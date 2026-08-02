// same_l_int8fused_cpu_amx.cpp — torch-free C++ AMX INT8 (w8a8) SAME-L decoder, FUSED all-integer.
//
// FUSED sibling of ../same_l_int8_cpu_amx (naive per-GEMM Q/DQ). Same 12-block/dim1536/24-head
// architecture, banded SWA diff-attn (+-17), global RoPE, sin-gate FF blocks 5..11, Linear(1536->512)
// map — same int8 weights (reused byte-identically) + same requant grid + same fp32 attention island.
// The ONLY change is the dataflow: it COPIES the DiT fused-epilogue design (dit_cpu_amx.cpp).
//
//   naive : dyt->fp32 ; quant->i8 ; GEMM ; deq->fp32 ; addbias ; act ; quant->i8 ; ...   (Q/DQ per GEMM)
//   FUSED : dyt_q (norm+quant->i8 ONE pass) ; GEMM ; deqglu_q (deq+bias+GLU+quant->i8 ONE pass) ;
//           GEMM ; deq_res (deq+bias+residual ONE pass).  Activation STAYS int8 between GLU & proj.
//
// Only standalone Q/DQ left = the 2 around the fp32 attention island. Residual stream x stays fp32
// (exactly like the DiT xnext / the bf16 engine) -> requant grid IDENTICAL to naive -> quality matches.
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            same_l_int8fused_cpu_amx.cpp -o same_l_int8fused_cpu_amx.so \
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

// ── architecture constants (same_l_decoder_torch.py) ──
static const int LAT=256, DIM=1536, H=24, HD=64, RD=32, HALF=16;
static const int NB=12, FF=4608, GLU2=9216, QKV=7680, OUTCH=512;
static const int SUB=17, SIN=16;
static const int SIN_START=5;                  // blocks >=5 use sin(pi*gate) gate
static const int BAND=17;
static const float SCALE=0.125f;

// ── optional phase profiler (SAMEL_PROF=1) ──
static double PROF[8]={0}; static long PROFN=0;
static const char* PROFLBL[8]={"gemm","attn","dyt_q","rope","deqglu","quant","deq_res",""};
static bool PROF_ON=false;
static inline double wt(){return omp_get_wtime();}
#define TB(id) do{ if(PROF_ON){double _n=wt(); PROF[id]+=_n-_pt; _pt=_n;} }while(0)

// ── fast vectorizable transcendentals ──
static inline float vexp(float x){
    x = x<-87.0f?-87.0f:(x>88.0f?88.0f:x);
    float z = x*1.442695041f;
    float n = std::floor(z+0.5f);
    float f = z-n;
    float p = 1.0f+f*(0.6931472f+f*(0.2402265f+f*(0.0555041f+f*(0.0096181f+f*0.0013333f))));
    uint32_t bits=(uint32_t)(((int)n+127)<<23); float s; std::memcpy(&s,&bits,4);
    return p*s;
}
static inline float vtanh(float x){ return 1.0f-2.0f/(vexp(2.0f*x)+1.0f); }
static inline float vsilu(float x){ return x/(1.0f+vexp(-x)); }
static inline float vsinpi(float g){
    float k = std::floor(g+0.5f);
    float r = g - k;
    float s = 1.0f - 2.0f*(float)(((long)k)&1L);
    float y = 3.14159265358979f*r;
    float y2 = y*y;
    float p = y*(1.0f + y2*(-0.16666667f + y2*(0.00833333f + y2*(-0.00019841f + y2*2.75573e-6f))));
    return s*p;
}
static inline int cdiv(int a,int b){return (a+b-1)/b;}
static inline int8_t q127(float v){ v=std::rintf(v); return (int8_t)(v>127.0f?127.0f:(v<-127.0f?-127.0f:v)); }

// ------------------------- mmap weights.bin + manifest -------------------------
struct Ten{void* p; std::string dt; long n; std::vector<long> shp;};
static std::map<std::string,Ten> TEN;
static char* BASE=nullptr;
static std::string WBASE="/weka2/cj/clod/same_l_int8fused_cpu_amx/weights";
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
    printf("[samel_i8fused] weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
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

// ------------------------- GLOBAL RoPE table (growable) -------------------------
static float RINV[HALF];
static std::vector<float> RCOS, RSIN;
static int RCAP=0;
static void rope_invfreq(){ for(int i=0;i<HALF;i++) RINV[i]=std::pow(10000.0f,-(float)(2*i)/(float)RD); }
static void rope_fill(int lo,int hi){
    for(int p=lo;p<hi;p++) for(int i=0;i<HALF;i++){
        RCOS[(size_t)p*HALF+i]=std::cos(p*RINV[i]); RSIN[(size_t)p*HALF+i]=std::sin(p*RINV[i]);
    }
}
static void rope_reserve(int cap){
    if(cap<=RCAP) return;
    RCOS.resize((size_t)cap*HALF); RSIN.resize((size_t)cap*HALF);
    rope_fill(RCAP,cap); RCAP=cap;
}
static void ensure_rope(int S){
    if(S<=RCAP) return;
    #pragma omp critical(rope_grow)
    { if(S>RCAP){ int nc=S+S/4; RCOS.resize((size_t)nc*HALF); RSIN.resize((size_t)nc*HALF);
                  rope_fill(RCAP,nc); RCAP=nc; } }
}

// ------------------------- reusable arena (per worker slot) -------------------------
// vs naive: dropped the [S,GLU2] fp32 `glu` buffer (deqglu_q reads int32 acc -> writes int8).
struct Arena{
    int cap=0;
    std::vector<float>   x,h,qkv,ao;              // f32 activations (residual stream stays fp32)
    std::vector<int8_t>  srcq;                     // int8 GEMM-src staging (>= max K = FF)
    std::vector<int32_t> acc;                      // int32 GEMM accumulator (>= max N = GLU2)
    std::vector<float>   ascl;                      // per-row activation scale [S]
    void ensure(int M){
        if(M<=cap) return; cap=M;
        x.assign((size_t)M*DIM,0); h.assign((size_t)M*DIM,0);
        qkv.assign((size_t)M*QKV,0); ao.assign((size_t)M*DIM,0);
        srcq.assign((size_t)M*GLU2,0);            // headroom (widest src is FF<GLU2)
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
        const float* cs=&RCOS[(size_t)m*HALF]; const float* sn=&RSIN[(size_t)m*HALF];
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
static void deq(const int32_t*acc,const float*as,const float*bs,int M,int N,float* o,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float am=as[m]; const int32_t* ar=acc+(size_t)m*N; float* orr=o+(size_t)m*N;
        #pragma omp simd
        for(int n=0;n<N;n++) orr[n]=(float)ar[n]*am*bs[n];
    }
}

// ===================== FUSED epilogues (COPY of the DiT rmsmodq / deqglu / deqgate semantics) ======
// dyt_q: FUSED DyT-norm + per-row symmetric int8 quant. Bit-identical to naive dyt+quant_rows, 1 pass.
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
// deqglu_q: FUSED dequant + bias + GLU(value*act(gate)) + per-row int8 quant. STAYS int8 into proj.
//   act = silu (blocks<5) or sin(pi*.) (blocks>=5). Replaces naive deq+addbias+glu(->fp32)+quant.
static void deqglu_q(const int32_t*acc,const float*as,const float*bs,const float*bias,
                     int M,int Fdim,bool use_sin,int8_t* q,float* s,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        float am=as[m]; const int32_t* ar=acc+(size_t)m*(2*Fdim); int8_t* qo=q+(size_t)m*Fdim;
        float hv[FF]; float amax=0.0f;
        if(use_sin){
            #pragma omp simd reduction(max:amax)
            for(int j=0;j<Fdim;j++){
                float val =(float)ar[j]*am*bs[j]           + bias[j];
                float gate=(float)ar[Fdim+j]*am*bs[Fdim+j] + bias[Fdim+j];
                float u=val*vsinpi(gate); hv[j]=u; float au=std::fabs(u); if(au>amax)amax=au;
            }
        }else{
            #pragma omp simd reduction(max:amax)
            for(int j=0;j<Fdim;j++){
                float val =(float)ar[j]*am*bs[j]           + bias[j];
                float gate=(float)ar[Fdim+j]*am*bs[Fdim+j] + bias[Fdim+j];
                float u=val*vsilu(gate); hv[j]=u; float au=std::fabs(u); if(au>amax)amax=au;
            }
        }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; s[m]=sc; float inv=1.0f/sc;
        #pragma omp simd
        for(int j=0;j<Fdim;j++) qo[j]=q127(hv[j]*inv);
    }
}
// deq_res: FUSED dequant + optional bias + residual add (in place: x += acc*as*bs + bias).
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

// ------------------------- BANDED differential attention (fp32, IDENTICAL to bf16/naive engine) --------
static void diff_attn_banded(const float*qkv,int S,float* out,bool par){
    #pragma omp parallel for schedule(dynamic) if(par)
    for(int hh=0;hh<H;hh++){
        static thread_local std::vector<float> G;
        G.resize((size_t)5*S*HD);
        float* qg=G.data(); float* kg=qg+(size_t)S*HD; float* vg=kg+(size_t)S*HD;
        float* qdg=vg+(size_t)S*HD; float* kdg=qdg+(size_t)S*HD;
        for(int t=0;t<S;t++){
            const float* r=qkv+(size_t)t*QKV + hh*HD;
            std::memcpy(qg +(size_t)t*HD, r+0*DIM, HD*4); std::memcpy(kg +(size_t)t*HD, r+1*DIM, HD*4);
            std::memcpy(vg +(size_t)t*HD, r+2*DIM, HD*4); std::memcpy(qdg+(size_t)t*HD, r+3*DIM, HD*4);
            std::memcpy(kdg+(size_t)t*HD, r+4*DIM, HD*4);
        }
        float sm[64], sd[64];
        for(int i=0;i<S;i++){
            int j0=std::max(0,i-BAND), j1=std::min(S-1,i+BAND);
            const float* qi=qg+(size_t)i*HD; const float* qdi=qdg+(size_t)i*HD;
            float mm=-1e30f, md=-1e30f;
            for(int j=j0;j<=j1;j++){
                const float* kj=kg+(size_t)j*HD; const float* kdj=kdg+(size_t)j*HD;
                float dm=0,dd=0;
                #pragma omp simd reduction(+:dm,dd)
                for(int d=0;d<HD;d++){ dm+=qi[d]*kj[d]; dd+=qdi[d]*kdj[d]; }
                dm*=SCALE; dd*=SCALE; sm[j-j0]=dm; sd[j-j0]=dd;
                if(dm>mm)mm=dm; if(dd>md)md=dd;
            }
            int n=j1-j0+1; float zm=0,zd=0;
            #pragma omp simd reduction(+:zm,zd)
            for(int j=0;j<n;j++){ sm[j]=vexp(sm[j]-mm); zm+=sm[j]; sd[j]=vexp(sd[j]-md); zd+=sd[j]; }
            float izm=1.0f/zm, izd=1.0f/zd;
            float om[HD], od[HD];
            for(int d=0;d<HD;d++){ om[d]=0; od[d]=0; }
            for(int j=j0;j<=j1;j++){
                const float* vj=vg+(size_t)j*HD; float wm=sm[j-j0]*izm, wd=sd[j-j0]*izd;
                #pragma omp simd
                for(int d=0;d<HD;d++){ om[d]+=wm*vj[d]; od[d]+=wd*vj[d]; }
            }
            float* orr=out+(size_t)i*DIM + hh*HD;
            #pragma omp simd
            for(int d=0;d<HD;d++) orr[d]=om[d]-od[d];
        }
    }
}

// ------------------------- one whole decode (latent[256,T] -> patches[512,16T]) ---------
static void decode(int slot,bool par,const float* latent,int T,float* out_patches){
    Arena& A=AR[slot];
    int S=SUB*T;                    // internal tokens (17T)
    A.ensure(S);
    ensure_rope(S);
    float* x=A.x.data(); float* h=A.h.data();
    float* qkv=A.qkv.data(); float* ao=A.ao.data();
    int8_t* srcq=A.srcq.data(); float* ascl=A.ascl.data(); int32_t* acc=A.acc.data();
    float rstd=F32("running_std")[0];

    // project_in (w8a8): f32 (latent^T*running_std) rows -> per-row quant -> GEMM -> deq+bias -> h[T,1536]
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        float row[LAT]; float amax=0.0f;
        for(int c=0;c<LAT;c++){ float v=latent[(size_t)c*T+t]*rstd; row[c]=v; float av=std::fabs(v); if(av>amax)amax=av; }
        float sc=amax>0.0f? amax/127.0f : 1e-12f; ascl[t]=sc; float inv=1.0f/sc;
        int8_t* qo=srcq+(size_t)t*LAT;
        for(int c=0;c<LAT;c++) qo[c]=q127(row[c]*inv);
    }
    gemm_i8(slot,srcq,I8("project_in.q"),T,DIM,LAT,acc);
    { const float* bs=F32("project_in.scale"); const float* bias=F32("project_in.b");
      #pragma omp parallel for schedule(static) if(par)
      for(int t=0;t<T;t++){ float am=ascl[t]; const int32_t* ar=acc+(size_t)t*DIM; float* o=h+(size_t)t*DIM;
        #pragma omp simd
        for(int n=0;n<DIM;n++) o[n]=(float)ar[n]*am*bs[n]+bias[n]; } }
    // expand: token t*17+0 = h[t]; t*17+1..16 = new_tokens
    const float* ntk=F32("new_tokens");
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        std::memcpy(x+(size_t)(t*SUB)*DIM, h+(size_t)t*DIM, DIM*sizeof(float));
        for(int s=1;s<SUB;s++) std::memcpy(x+(size_t)(t*SUB+s)*DIM, ntk, DIM*sizeof(float));
    }

    auto run_block=[&](int b){
        double _pt=wt();
        bool use_sin=(b>=SIN_START);
        char pb[8]; snprintf(pb,sizeof pb,"b%d.",b);
        auto W=[&](const char*n){return std::string(pb)+n;};
        // ---- attention ----
        dyt_q(x,S,DIM,F32(W("pre.alpha"))[0],F32(W("pre.gamma")),F32(W("pre.beta")),srcq,ascl,par); TB(2);
        gemm_i8(slot,srcq,I8(W("qkv.q")),S,QKV,DIM,acc); TB(0);
        deq(acc,ascl,F32(W("qkv.scale")),S,QKV,qkv,par); TB(6);           // the ONE dequant into the fp32 island
        dyt_slice(qkv,S,QKV,0*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,S,QKV,3*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,S,QKV,1*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par);
        dyt_slice(qkv,S,QKV,4*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par); TB(2);
        rope_slice(qkv,S,QKV,0*DIM,par); rope_slice(qkv,S,QKV,1*DIM,par);
        rope_slice(qkv,S,QKV,3*DIM,par); rope_slice(qkv,S,QKV,4*DIM,par); TB(3);
        diff_attn_banded(qkv,S,ao,par); TB(1);                            // fp32 attention island
        quant_rows(ao,S,DIM,srcq,ascl,par); TB(5);                        // the ONE quant out of the island
        gemm_i8(slot,srcq,I8(W("out.q")),S,DIM,DIM,acc); TB(0);
        deq_res(acc,ascl,F32(W("out.scale")),nullptr,x,S,DIM,par); TB(6); // FUSED deq + residual (x += .)
        // ---- FFN ----
        dyt_q(x,S,DIM,F32(W("ff.alpha"))[0],F32(W("ff.gamma")),F32(W("ff.beta")),srcq,ascl,par); TB(2);
        gemm_i8(slot,srcq,I8(W("glu.q")),S,GLU2,DIM,acc); TB(0);
        deqglu_q(acc,ascl,F32(W("glu.scale")),F32(W("glu.b")),S,FF,use_sin,srcq,ascl,par); TB(4); // deq+bias+GLU+quant->i8 (STAYS i8)
        gemm_i8(slot,srcq,I8(W("proj.q")),S,DIM,FF,acc); TB(0);           // int8 GEMM directly on int8
        deq_res(acc,ascl,F32(W("proj.scale")),F32(W("proj.b")),x,S,DIM,par); TB(6); // FUSED deq + bias + residual
    };

    for(int b=0;b<NB;b++) run_block(b);

    // drop latent slot 0 -> y[16T,DIM]
    int L=SIN*T;
    float* y=A.h.data();
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++)
        for(int s=1;s<SUB;s++)
            std::memcpy(y+(size_t)(t*SIN+(s-1))*DIM, x+(size_t)(t*SUB+s)*DIM, DIM*sizeof(float));
    // output map: plain Linear(1536->512) w8a8 -> [L,512] ; fused deq+bias+transpose -> out_patches[512,L]
    quant_rows(y,L,DIM,srcq,ascl,par);
    gemm_i8(slot,srcq,I8("map.q"),L,OUTCH,DIM,acc);
    { const float* bs=F32("map.scale"); const float* bias=F32("map.b");
      #pragma omp parallel for schedule(static) if(par)
      for(int l=0;l<L;l++){ float am=ascl[l]; const int32_t* ar=acc+(size_t)l*OUTCH;
        for(int ch=0;ch<OUTCH;ch++) out_patches[(size_t)ch*L+l]=(float)ar[ch]*am*bs[ch]+bias[ch]; } }
}

// ============================== C ABI ==============================
static int NTHREADS=16;
extern "C" {

int samel_init(const char* weights_base,int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    NTHREADS=threads;
    if(weights_base && weights_base[0]) WBASE=weights_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[samel_i8fused] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    omp_set_max_active_levels(1);
    #pragma omp parallel num_threads(threads)
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init(); mmcache_init(threads);
    load_weights(); rope_invfreq();
    rope_reserve(SUB*1024);
    AR.resize(threads+1);
    PROF_ON = getenv("SAMEL_PROF")!=nullptr;
    printf("[samel_i8fused] init ok: threads=%d isa=%s\n",threads,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));
    fflush(stdout);
    return 0;
}

void samel_forward(const float* latent,int T,float* out_patches){
    decode(0,true,latent,T,out_patches);
}

void samel_unpatch(const float* patches,int L,float* pcm){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int st=0;st<2;st++)for(int l=0;l<L;l++){
        float* o=pcm+((size_t)st*L+l)*256;
        for(int p=0;p<256;p++) o[p]=patches[((size_t)st*256+p)*L+l];
    }
}

int samel_DIM(){return DIM;} int samel_SUB(){return SUB;} int samel_SIN(){return SIN;}

void samel_prof_dump(){
    double tot=0; for(int i=0;i<7;i++) tot+=PROF[i];
    printf("[prof] ");
    for(int i=0;i<7;i++) printf("%s=%.1fms(%.0f%%) ",PROFLBL[i],PROF[i]*1e3,tot>0?100*PROF[i]/tot:0);
    printf(" total=%.1fms\n",tot*1e3);
    for(int i=0;i<8;i++) PROF[i]=0; PROFN=0; fflush(stdout);
}

} // extern "C"

// ---- chunk-parallel decode (PRIMARY path) appended below via include ----
#include "same_l_int8fused_chunk.inc"
