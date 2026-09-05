// same_s_int8fused_cpu_amx.cpp — torch-free C++ AMX INT8 (w8a8) SAME-S decoder, FUSED all-integer.
//
// FUSED sibling of ../same_s_int8_cpu_amx (naive per-GEMM Q/DQ). Same architecture, same int8
// weights (reused byte-identically) + same per-row/per-channel requant grid, and the same fp32
// differential attention island. The ONLY change is the dataflow around the int8 GEMMs — it COPIES
// the DiT engine's fused-epilogue design (dit_cpu_amx.cpp: rmsmodq / deqglu / deqgate):
//
//   naive : dyt->fp32 ; quant->i8 ; GEMM ; deq->fp32 ; addbias ; act ; quant->i8 ; ...   (Q/DQ per GEMM)
//   FUSED : dyt_q (norm+quant->i8 ONE pass) ; GEMM ; deqglu_q (deq+bias+GLU+quant->i8 ONE pass) ;
//           GEMM ; deq_res (deq+bias+residual ONE pass).  Activation STAYS int8 between GLU & proj.
//
// The ONLY standalone Q/DQ left is the 2 around the mandatory fp32 attention island (deq(qkv)->fp32
// in, quant(attn-out)->i8 out) — every other requant is folded into a norm / GLU / residual pass
// that already existed. Residual stream x stays fp32 (exactly like the DiT xnext / the bf16 engine),
// so the requant grid is IDENTICAL to naive int8 -> quality must match (verified).
//
//   sames_init(weights_base, threads)  -> mmap int8 weights ONCE + AMX/omp/oneDNN
//   sames_forward(latent[1,256,T], T, out_patches[1,512,16T])   -> whole decode
//   sames_forward_chunked(latent, T, C, overlap, parallel, out) -> cache-blocked decode
//   sames_unpatch(patches[1,512,L], L, pcm[1,2,256L])           -> torch-free unpatch
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            same_s_int8fused_cpu_amx.cpp -o same_s_int8fused_cpu_amx.so \
//            -L$ONELIB -ldnnl -Wl,-rpath,$ONELIB -ldl -lpthread -lm
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <omp.h>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

// ── architecture constants (same_s_decoder_torch.py) ──
static const int LAT=256, DIM=768, H=12, HD=64, RD=32, HALF=16;
static const int NB=6, FF=2304, GLU2=4608, QKV=3840, OUTCH=512;
static const int SUB=17, ECH=34, SHIFT=17, SIN=16;      // 17 tok/latent, 34-tok chunk
static const float SCALE=0.125f;                         // HD**-0.5 = 64**-0.5

// ── optional phase profiler (SAMES_PROF=1) ──
static double PROF[8]={0}; static long PROFN=0;
static const char* PROFLBL[8]={"gemm","attn","dyt_q","rope","deqglu","quant","deq_res",""};
static bool PROF_ON=false;
static inline double wt(){return omp_get_wtime();}
#define TB(id) do{ if(PROF_ON){double _n=wt(); PROF[id]+=_n-_pt; _pt=_n;} }while(0)

// ── fast vectorizable transcendentals (pure float, no libm/branches -> AVX-512 auto-vec) ──
static inline float vexp(float x){
    x = x<-87.0f?-87.0f:(x>88.0f?88.0f:x);
    float z = x*1.442695041f;
    float n = std::floor(z+0.5f);
    float f = z-n;
    float p = 1.0f+f*(0.6931472f+f*(0.2402265f+f*(0.0555041f+f*(0.0096181f+f*0.0013333f))));
    uint32_t bits=(uint32_t)(((int)n+127)<<23); float s; std::memcpy(&s,&bits,4);
    return p*s;
}
static inline float vtanh(float x){ return 1.0f-2.0f/(vexp(2.0f*x)+1.0f); }   // == Triton DyT kernel
static inline float vsilu(float x){ return x/(1.0f+vexp(-x)); }
static inline int cdiv(int a,int b){return (a+b-1)/b;}
static inline int8_t q127(float v){ v=std::rintf(v); return (int8_t)(v>127.0f?127.0f:(v<-127.0f?-127.0f:v)); }

// ------------------------- mmap weights.bin + manifest -------------------------
struct Ten{void* p; std::string dt; long n; std::vector<long> shp;};
static std::map<std::string,Ten> TEN;
static char* BASE=nullptr;

// Engine paths resolve from $SA3_CPUAMX_HOME (same base the Python side uses), so nothing
// absolute is baked into the binary. Falls back to the current directory.
static const char* sa3_home() {
    const char* v = getenv("SA3_CPUAMX_HOME");
    return (v && *v) ? v : ".";
}
static std::string WBASE = std::string(sa3_home()) + "/same_s_int8fused_cpu_amx/weights";
static void load_weights(){
    std::string bin=WBASE+".bin";
    int fd=open(bin.c_str(),O_RDONLY); struct stat st; fstat(fd,&st);
    BASE=(char*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(BASE==MAP_FAILED){perror("mmap");exit(1);} close(fd);
    std::ifstream mf(WBASE+"_manifest.txt"); std::string line;
    while(std::getline(mf,line)){
        std::istringstream ss(line); Ten t; std::string name; long off;
        ss>>name>>t.dt>>off>>t.n; long d; while(ss>>d)t.shp.push_back(d);
        t.p=(void*)(BASE+off); TEN[name]=t;
    }
    printf("[sames_i8fused] weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
}
static float*  F32(const std::string&k){return (float*)TEN.at(k).p;}
static int8_t* I8 (const std::string&k){return (int8_t*)TEN.at(k).p;}

// ------------------------- oneDNN int8 matmul (s8 x s8 -> s32): primitive+handle cache
static dnnl::engine* ENG=nullptr;
static void onednn_init(){ ENG=new dnnl::engine(dnnl::engine::kind::cpu,0); }
struct MMKey{int M,N,K; bool operator<(const MMKey&o)const{
    return M!=o.M?M<o.M:(N!=o.N?N<o.N:K<o.K);}};
struct MMEnt{dnnl::matmul prim; dnnl::memory am,bm,cm;};
struct MMCache{ std::map<MMKey,MMEnt> mm; dnnl::stream* strm=nullptr; };
static std::vector<MMCache> MMC;
static void mmcache_init(int nworkers){
    MMC.clear(); MMC.resize(std::max(1,nworkers)+1);
    for(auto& c:MMC) c.strm=new dnnl::stream(*ENG);
}
static void gemm_i8(int slot,const int8_t*A,const int8_t*B,int M,int N,int K,int32_t* C){
    using dt=dnnl::memory::data_type;
    MMCache& c=MMC[slot];
    MMKey key{M,N,K}; auto it=c.mm.find(key);
    if(it==c.mm.end()){
        #pragma omp critical(mmcreate)
        {
          it=c.mm.find(key);
          if(it==c.mm.end()){
            dnnl::memory::desc a_md({M,K},dt::s8,{K,1});
            dnnl::memory::desc b_md({K,N},dt::s8,{N,1});
            dnnl::memory::desc c_md({M,N},dt::s32,{N,1});
            dnnl::matmul::primitive_desc pd(*ENG,a_md,b_md,c_md);
            MMEnt e{dnnl::matmul(pd),
                    dnnl::memory(pd.src_desc(),*ENG,(void*)A),
                    dnnl::memory(pd.weights_desc(),*ENG,(void*)B),
                    dnnl::memory(pd.dst_desc(),*ENG,(void*)C)};
            it=c.mm.emplace(key,std::move(e)).first;
          }
        }
    }
    MMEnt& e=it->second;
    e.am.set_data_handle((void*)A); e.bm.set_data_handle((void*)B); e.cm.set_data_handle((void*)C);
    e.prim.execute(*c.strm,{{DNNL_ARG_SRC,e.am},{DNNL_ARG_WEIGHTS,e.bm},{DNNL_ARG_DST,e.cm}});
    c.strm->wait();
}

// ------------------------- RoPE table for one 34-token chunk (positions 0..33) -------------------------
static float RCOS[ECH*HALF], RSIN[ECH*HALF];   // [pos, i] i=0..15
static void build_rope(){
    for(int i=0;i<HALF;i++){
        float inv=std::pow(10000.0f,-(float)(2*i)/(float)RD);
        for(int p=0;p<ECH;p++){ RCOS[p*HALF+i]=std::cos(p*inv); RSIN[p*HALF+i]=std::sin(p*inv); }
    }
}

// ------------------------- reusable arena (per max-M, per worker slot) -------------------------
// vs naive: dropped the [M,GLU2] fp32 `glu` buffer entirely (deqglu_q reads int32 acc -> writes int8).
struct Arena{
    int cap=0;
    std::vector<float>   x,xt,h,qkv,ao;          // f32 activations (residual stream stays fp32)
    std::vector<int8_t>  srcq;                    // int8 GEMM-src staging (>= max K = DIM*3 = FF)
    std::vector<int32_t> acc;                     // int32 GEMM accumulator (>= max N = GLU2)
    std::vector<float>   ascl;                     // per-row activation scale [M]
    void ensure(int M){
        if(M<=cap) return; cap=M;
        x.assign((size_t)M*DIM,0); xt.assign((size_t)M*DIM,0); h.assign((size_t)M*DIM,0);
        qkv.assign((size_t)M*QKV,0); ao.assign((size_t)M*DIM,0);
        srcq.assign((size_t)M*GLU2,0);            // headroom (widest src is DIM*3=FF<GLU2)
        acc.assign((size_t)M*GLU2,0);             // widest N is glu_proj = GLU2
        ascl.assign((size_t)M,0);
    }
};
static std::vector<Arena> AR;

// ------------------------- fp32 attention-prep elementwise (identical to bf16/naive engine) -------------------------
static void dyt_slice(float*base,int M,int W,int coloff,float alpha,const float*g,const float*b,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float* r=base+(size_t)m*W+coloff;
        for(int hh=0;hh<H;hh++){
            float* rh=r+hh*HD;
            #pragma omp simd
            for(int d=0;d<HD;d++) rh[d]=g[d]*vtanh(alpha*rh[d])+b[d];
        }
    }
}
static void rope_slice(float*base,int M,int W,int coloff,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        int p=m%ECH; const float* cs=&RCOS[p*HALF]; const float* sn=&RSIN[p*HALF];
        float* r=base+(size_t)m*W+coloff;
        for(int hh=0;hh<H;hh++){
            float* hd=r+hh*HD;
            for(int i=0;i<HALF;i++){
                float a=hd[i], b=hd[i+HALF], c=cs[i], s=sn[i];
                hd[i]=a*c-b*s; hd[i+HALF]=b*c+a*s;
            }
        }
    }
}

// ------------------------- int8 GEMM glue -------------------------
// per-row (per-token) dynamic symmetric int8 (mandatory ao->i8 boundary + project_in/conv)
static void quant_rows(const float*x,int M,int K,int8_t* q,float* s,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        const float* r=x+(size_t)m*K; int8_t* qo=q+(size_t)m*K;
        float amax=0.0f;
        #pragma omp simd reduction(max:amax)
        for(int k=0;k<K;k++){ float av=std::fabs(r[k]); if(av>amax)amax=av; }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; s[m]=sc; float inv=1.0f/sc;
        #pragma omp simd
        for(int k=0;k<K;k++) qo[k]=q127(r[k]*inv);
    }
}
// dequant s32 acc -> f32 (the ONE dequant into the fp32 attention island): o=acc*a_scale[m]*w_scale[n]
static void deq(const int32_t*acc,const float*as,const float*bs,int M,int N,float* o,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float am=as[m]; const int32_t* ar=acc+(size_t)m*N; float* orr=o+(size_t)m*N;
        #pragma omp simd
        for(int n=0;n<N;n++) orr[n]=(float)ar[n]*am*bs[n];
    }
}

// ===================== FUSED epilogues (COPY of the DiT rmsmodq / deqglu / deqgate semantics) ======
// dyt_q: FUSED DyT-norm + per-row symmetric int8 quant.  o_i8[m,j]=q(gamma[j]*tanh(alpha*x)+beta[j]).
//   Replaces naive dyt(->fp32 h) + quant_rows(h). Bit-identical (t is fp32 either way), 1 pass not 2.
static void dyt_q(const float*x,int M,int K,float alpha,const float*g,const float*b,
                  int8_t* q,float* s,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        const float* xr=x+(size_t)m*K; int8_t* qo=q+(size_t)m*K;
        float t[DIM]; float amax=0.0f;
        #pragma omp simd reduction(max:amax)
        for(int j=0;j<K;j++){ float u=g[j]*vtanh(alpha*xr[j])+b[j]; t[j]=u; float au=std::fabs(u); if(au>amax)amax=au; }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; s[m]=sc; float inv=1.0f/sc;
        #pragma omp simd
        for(int j=0;j<K;j++) qo[j]=q127(t[j]*inv);
    }
}
// deqglu_q: FUSED dequant + bias + GLU(value*silu(gate)) + per-row int8 quant.  STAYS int8 into proj.
//   acc[M,2F] int32, as[M] (glu-src row scale), bs[2F] (glu w per-chan scale), bias[2F].
//   Replaces naive deq(->fp32 [M,2F]) + addbias + glu-silu(->fp32 [M,F]) + quant. 1 pass, no fp32 glu buffer.
static void deqglu_q(const int32_t*acc,const float*as,const float*bs,const float*bias,
                     int M,int Fdim,int8_t* q,float* s,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float am=as[m]; const int32_t* ar=acc+(size_t)m*(2*Fdim); int8_t* qo=q+(size_t)m*Fdim;
        float hv[FF]; float amax=0.0f;
        #pragma omp simd reduction(max:amax)
        for(int j=0;j<Fdim;j++){
            float val =(float)ar[j]*am*bs[j]         + bias[j];
            float gate=(float)ar[Fdim+j]*am*bs[Fdim+j] + bias[Fdim+j];
            float u=val*vsilu(gate); hv[j]=u; float au=std::fabs(u); if(au>amax)amax=au;
        }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; s[m]=sc; float inv=1.0f/sc;
        #pragma omp simd
        for(int j=0;j<Fdim;j++) qo[j]=q127(hv[j]*inv);
    }
}
// deq_res: FUSED dequant + optional bias + residual add (in place: x += acc*as*bs + bias).
//   Replaces naive deq(->fp32 h) + addbias + resadd. 1 pass, no fp32 h buffer. Residual stays fp32.
static void deq_res(const int32_t*acc,const float*as,const float*bs,const float*bias,
                    float* x,int M,int N,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float am=as[m]; const int32_t* ar=acc+(size_t)m*N; float* xr=x+(size_t)m*N;
        if(bias){
            #pragma omp simd
            for(int n=0;n<N;n++) xr[n]+=(float)ar[n]*am*bs[n]+bias[n];
        }else{
            #pragma omp simd
            for(int n=0;n<N;n++) xr[n]+=(float)ar[n]*am*bs[n];
        }
    }
}

// fp32 differential attention over the nc 34-token chunks, H heads (IDENTICAL to bf16/naive engine).
static void diff_attn(const float*qkv,int M,float* out,bool par){
    int nc=M/ECH;
    #pragma omp parallel for collapse(2) schedule(static) if(par)
    for(int c=0;c<nc;c++)for(int hh=0;hh<H;hh++){
        const int base=c*ECH;
        float qg[ECH*HD],kg[ECH*HD],vg[ECH*HD],qdg[ECH*HD],kdg[ECH*HD];
        for(int t=0;t<ECH;t++){
            const float* r=qkv+(size_t)(base+t)*QKV + hh*HD;
            std::memcpy(qg +t*HD, r+0*DIM, HD*4); std::memcpy(kg +t*HD, r+1*DIM, HD*4);
            std::memcpy(vg +t*HD, r+2*DIM, HD*4); std::memcpy(qdg+t*HD, r+3*DIM, HD*4);
            std::memcpy(kdg+t*HD, r+4*DIM, HD*4);
        }
        for(int i=0;i<ECH;i++){
            const float* qi=qg+i*HD; const float* qdi=qdg+i*HD;
            float sm[ECH], sd[ECH]; float mm=-1e30f, md=-1e30f;
            for(int j=0;j<ECH;j++){
                const float* kj=kg+j*HD; const float* kdj=kdg+j*HD;
                float dm=0,dd=0;
                #pragma omp simd reduction(+:dm,dd)
                for(int d=0;d<HD;d++){ dm+=qi[d]*kj[d]; dd+=qdi[d]*kdj[d]; }
                dm*=SCALE; dd*=SCALE; sm[j]=dm; sd[j]=dd;
                if(dm>mm)mm=dm; if(dd>md)md=dd;
            }
            float zm=0,zd=0;
            #pragma omp simd reduction(+:zm,zd)
            for(int j=0;j<ECH;j++){ sm[j]=vexp(sm[j]-mm); zm+=sm[j]; sd[j]=vexp(sd[j]-md); zd+=sd[j]; }
            float izm=1.0f/zm, izd=1.0f/zd;
            float om[HD], od[HD];
            for(int d=0;d<HD;d++){ om[d]=0; od[d]=0; }
            for(int j=0;j<ECH;j++){
                const float* vj=vg+j*HD; float wm=sm[j]*izm, wd=sd[j]*izd;
                #pragma omp simd
                for(int d=0;d<HD;d++){ om[d]+=wm*vj[d]; od[d]+=wd*vj[d]; }
            }
            float* orr=out+(size_t)(base+i)*DIM + hh*HD;
            #pragma omp simd
            for(int d=0;d<HD;d++) orr[d]=om[d]-od[d];
        }
    }
}

// ------------------------- one whole decode (latent[256,T] -> patches[512,16T]) ---------
static void decode(int slot,bool par,const float* latent,int T,float* out_patches){
    Arena& A=AR[slot];
    int iT=SUB*T;              // internal tokens after new-token expansion (17T)
    int M2=iT+ECH;             // padded token count for the shifted 2nd half
    A.ensure(M2);
    float* x=A.x.data(); float* xt=A.xt.data(); float* h=A.h.data();
    float* qkv=A.qkv.data(); float* ao=A.ao.data();
    int8_t* srcq=A.srcq.data(); float* ascl=A.ascl.data(); int32_t* acc=A.acc.data();
    float rstd=F32("running_std")[0];

    // project_in (w8a8): build f32 (latent^T*running_std) rows, per-row quant, GEMM, deq+bias -> xt[T,768]
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        float row[LAT]; float amax=0.0f;
        for(int c=0;c<LAT;c++){ float v=latent[(size_t)c*T+t]*rstd; row[c]=v; float av=std::fabs(v); if(av>amax)amax=av; }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; ascl[t]=sc; float inv=1.0f/sc;
        int8_t* qo=srcq+(size_t)t*LAT;
        for(int c=0;c<LAT;c++) qo[c]=q127(row[c]*inv);
    }
    gemm_i8(slot,srcq,I8("project_in.q"),T,DIM,LAT,acc);
    { const float* as=ascl; const float* bs=F32("project_in.scale"); const float* bias=F32("project_in.b");
      #pragma omp parallel for schedule(static) if(par)
      for(int t=0;t<T;t++){ float am=as[t]; const int32_t* ar=acc+(size_t)t*DIM; float* o=xt+(size_t)t*DIM;
        #pragma omp simd
        for(int n=0;n<DIM;n++) o[n]=(float)ar[n]*am*bs[n]+bias[n]; } }
    // expand: token t*17+0 = xt[t]; t*17+1..16 = new_tokens
    const float* ntk=F32("new_tokens");
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        std::memcpy(x+(size_t)(t*SUB)*DIM, xt+(size_t)t*DIM, DIM*sizeof(float));
        for(int s=1;s<SUB;s++) std::memcpy(x+(size_t)(t*SUB+s)*DIM, ntk, DIM*sizeof(float));
    }

    auto run_block=[&](int b,int M){
        double _pt=wt();
        char pb[8]; snprintf(pb,sizeof pb,"b%d.",b);
        auto W=[&](const char*n){return std::string(pb)+n;};
        // ---- attention ----
        // FUSED dyt_norm + quant -> int8 (was: dyt->fp32 h ; quant_rows h)
        dyt_q(x,M,DIM,F32(W("pre.alpha"))[0],F32(W("pre.gamma")),F32(W("pre.beta")),srcq,ascl,par); TB(2);
        gemm_i8(slot,srcq,I8(W("qkv.q")),M,QKV,DIM,acc); TB(0);
        deq(acc,ascl,F32(W("qkv.scale")),M,QKV,qkv,par); TB(6);          // the ONE dequant into the fp32 island
        dyt_slice(qkv,M,QKV,0*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,M,QKV,3*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,M,QKV,1*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par);
        dyt_slice(qkv,M,QKV,4*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par); TB(2);
        rope_slice(qkv,M,QKV,0*DIM,par); rope_slice(qkv,M,QKV,1*DIM,par);
        rope_slice(qkv,M,QKV,3*DIM,par); rope_slice(qkv,M,QKV,4*DIM,par); TB(3);
        diff_attn(qkv,M,ao,par); TB(1);                                   // fp32 attention island
        quant_rows(ao,M,DIM,srcq,ascl,par); TB(5);                        // the ONE quant out of the island
        gemm_i8(slot,srcq,I8(W("out.q")),M,DIM,DIM,acc); TB(0);
        deq_res(acc,ascl,F32(W("out.scale")),nullptr,x,M,DIM,par); TB(6); // FUSED deq + residual (x += .)
        // ---- FFN ----
        dyt_q(x,M,DIM,F32(W("ff.alpha"))[0],F32(W("ff.gamma")),F32(W("ff.beta")),srcq,ascl,par); TB(2);
        gemm_i8(slot,srcq,I8(W("glu.q")),M,GLU2,DIM,acc); TB(0);
        deqglu_q(acc,ascl,F32(W("glu.scale")),F32(W("glu.b")),M,FF,srcq,ascl,par); TB(4); // deq+bias+SiLU-GLU+quant->i8 (STAYS i8)
        gemm_i8(slot,srcq,I8(W("proj.q")),M,DIM,FF,acc); TB(0);           // int8 GEMM directly on int8 (no fp32 round-trip)
        deq_res(acc,ascl,F32(W("proj.scale")),F32(W("proj.b")),x,M,DIM,par); TB(6); // FUSED deq + bias + residual
    };

    // blocks 0..2 over the iT tokens
    for(int b=0;b<3;b++) run_block(b,iT);
    // midpoint shift by 17 (pure f32 memcpy, identical to bf16/naive engine)
    std::memcpy(xt, x, (size_t)SHIFT*DIM*sizeof(float));
    std::memcpy(xt+(size_t)SHIFT*DIM, x, (size_t)iT*DIM*sizeof(float));
    std::memcpy(xt+(size_t)(SHIFT+iT)*DIM, x+(size_t)(iT-SHIFT)*DIM, (size_t)SHIFT*DIM*sizeof(float));
    std::memcpy(x, xt, (size_t)M2*DIM*sizeof(float));
    // blocks 3..5 over M2 tokens
    for(int b=3;b<6;b++) run_block(b,M2);
    // strip shift + drop latent slot -> y[16T, DIM]
    int L=SIN*T;
    float* y=A.h.data();      // reuse h as [16T,DIM] staging
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        for(int s=1;s<SUB;s++)
            std::memcpy(y+(size_t)(t*SIN+(s-1))*DIM, x+(size_t)(SHIFT+t*SUB+s)*DIM, DIM*sizeof(float));
    }
    // conv1d 768->512 k3 pad1 via im2col (w8a8): fused im2col + per-row quant -> s8s8 GEMM -> fused deq+bias+transpose
    #pragma omp parallel for schedule(static) if(par)
    for(int l=0;l<L;l++){
        float col[DIM*3]; float amax=0.0f;
        for(int c=0;c<DIM;c++)for(int k=0;k<3;k++){
            int j=l+k-1;                       // pad=1
            float v=(j>=0&&j<L)? y[(size_t)j*DIM+c] : 0.0f;
            col[c*3+k]=v; float av=std::fabs(v); if(av>amax)amax=av;
        }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; ascl[l]=sc; float inv=1.0f/sc;
        int8_t* qo=srcq+(size_t)l*(DIM*3);
        for(int i=0;i<DIM*3;i++) qo[i]=q127(col[i]*inv);
    }
    gemm_i8(slot,srcq,I8("conv.q"),L,OUTCH,DIM*3,acc);
    // FUSED deq + bias + transpose  [L,512] -> out_patches[512,L]
    { const float* bs=F32("conv.scale"); const float* bias=F32("conv.b");
      #pragma omp parallel for schedule(static) if(par)
      for(int l=0;l<L;l++){ float am=ascl[l]; const int32_t* ar=acc+(size_t)l*OUTCH;
        for(int ch=0;ch<OUTCH;ch++) out_patches[(size_t)ch*L+l]=(float)ar[ch]*am*bs[ch]+bias[ch]; } }
}

// ============================== C ABI ==============================
static int NTHREADS=16;
extern "C" {

int sames_init(const char* weights_base,int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    NTHREADS=threads;
    if(weights_base && weights_base[0]) WBASE=weights_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[sames_i8fused] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    omp_set_max_active_levels(1);
    #pragma omp parallel num_threads(threads)
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init(); mmcache_init(threads);
    load_weights(); build_rope();
    AR.resize(threads+1);
    PROF_ON = getenv("SAMES_PROF")!=nullptr;
    printf("[sames_i8fused] init ok: threads=%d isa=%s\n",threads,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));
    fflush(stdout);
    return 0;
}

void sames_forward(const float* latent,int T,float* out_patches){
    decode(0,true,latent,T,out_patches);
}

void sames_unpatch(const float* patches,int L,float* pcm){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int st=0;st<2;st++)for(int l=0;l<L;l++){
        float* o=pcm+((size_t)st*L+l)*256;
        for(int p=0;p<256;p++) o[p]=patches[((size_t)st*256+p)*L+l];
    }
}

int sames_DIM(){return DIM;} int sames_SUB(){return SUB;} int sames_SIN(){return SIN;}

void sames_prof_dump(){
    double tot=0; for(int i=0;i<7;i++) tot+=PROF[i];
    printf("[prof] ");
    for(int i=0;i<7;i++) printf("%s=%.1fms(%.0f%%) ",PROFLBL[i],PROF[i]*1e3,tot>0?100*PROF[i]/tot:0);
    printf(" total=%.1fms\n",tot*1e3);
    for(int i=0;i<8;i++) PROF[i]=0; PROFN=0; fflush(stdout);
}

} // extern "C"

// ---- chunk-parallel decode appended below via include (shares all statics) ----
#include "same_s_int8fused_chunk.inc"
