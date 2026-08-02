// same_s_cpu_amx.cpp — torch-free C++ AMX-BF16 SAME-S decoder as a callable .so.
//
// Mirrors the proven int8 DiT engine (dit_cpu_amx.cpp) but for the bf16 SAME-S SHIP
// config: oneDNN AMX-BF16 GEMMs for every linear + the WNConv1d (im2col), and plain
// fp32 C++ for the cancellation-fragile elementwise (DyT norms, RoPE, GLU-SiLU) and the
// tiny 34-token differential attention (bf16 attn = 37.5 dB, below target -> keep fp32).
//
//   sames_init(weights_base, threads)  -> mmap bf16 weights ONCE + AMX/omp/oneDNN
//   sames_forward(latent[1,256,T], T, out_patches[1,512,16T])   -> whole decode
//   sames_forward_chunked(latent, T, C, overlap, parallel, out) -> cache-blocked decode
//   sames_unpatch(patches[1,512,L], L, pcm[1,2,256L])           -> torch-free unpatch
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            same_s_cpu_amx.cpp -o same_s_cpu_amx.so $ONELIB/libdnnl.a -ldl -lpthread -lm
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
static const char* PROFLBL[8]={"gemm","attn","dyt","rope","glu","cast","misc",""};
static bool PROF_ON=false;
static inline double wt(){return omp_get_wtime();}
#define TB(id) do{ if(PROF_ON){double _n=wt(); PROF[id]+=_n-_pt; _pt=_n;} }while(0)

typedef uint16_t bf16;
static inline bf16 f2b(float f){ // round-to-nearest-even f32->bf16
    uint32_t x; std::memcpy(&x,&f,4);
    uint32_t r=x+0x7fff+((x>>16)&1); return (bf16)(r>>16);
}
static inline float b2f(bf16 h){ uint32_t x=(uint32_t)h<<16; float f; std::memcpy(&f,&x,4); return f; }

// ── fast vectorizable transcendentals (pure float, no libm/branches -> AVX-512 auto-vec) ──
// exp: 2^f degree-5 minimax on [-0.5,0.5] + ldexp via exponent bits. ~2e-7 rel error.
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

// ------------------------- mmap weights.bin + manifest -------------------------
struct Ten{void* p; std::string dt; long n; std::vector<long> shp;};
static std::map<std::string,Ten> TEN;
static char* BASE=nullptr;
static std::string WBASE="/weka2/cj/clod/same_s_cpu_amx/weights";
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
    printf("[sames] weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
}
static float* F32(const std::string&k){return (float*)TEN.at(k).p;}
static bf16*  BF (const std::string&k){return (bf16*)TEN.at(k).p;}

// ------------------------- oneDNN bf16 matmul (bf16 x bf16 -> f32): primitive+handle cache
static dnnl::engine* ENG=nullptr;
static void onednn_init(){ ENG=new dnnl::engine(dnnl::engine::kind::cpu,0); }
struct MMKey{int M,N,K; bool operator<(const MMKey&o)const{
    return M!=o.M?M<o.M:(N!=o.N?N<o.N:K<o.K);}};
struct MMEnt{dnnl::matmul prim; dnnl::memory am,bm,cm;};

// Per-thread matmul cache (so parallel chunks don't fight over one primitive's memory
// handles). Index 0 = the shared/16-thread path; >0 = per-worker for chunk-parallel.
struct MMCache{ std::map<MMKey,MMEnt> mm; dnnl::stream* strm=nullptr; };
static std::vector<MMCache> MMC;
static void mmcache_init(int nworkers){
    MMC.clear(); MMC.resize(std::max(1,nworkers)+1);
    for(auto& c:MMC) c.strm=new dnnl::stream(*ENG);
}
// src A[M,K] bf16, wei B[K,N] bf16 (row-major), dst C[M,N] f32. cache slot `slot`.
static void gemm_bf16(int slot,const bf16*A,const bf16*B,int M,int N,int K,float* C){
    using dt=dnnl::memory::data_type;
    MMCache& c=MMC[slot];
    MMKey key{M,N,K}; auto it=c.mm.find(key);
    if(it==c.mm.end()){
        // primitive creation is rare (cached per shape); guard it so concurrent chunk
        // workers can build primitives from the shared engine safely. Execution stays lock-free.
        #pragma omp critical(mmcreate)
        {
          it=c.mm.find(key);
          if(it==c.mm.end()){
            dnnl::memory::desc a_md({M,K},dt::bf16,{K,1});
            dnnl::memory::desc b_md({K,N},dt::bf16,{N,1});
            dnnl::memory::desc c_md({M,N},dt::f32, {N,1});
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

// ------------------------- reusable arena (allocated per max-M, reused across blocks) -------------------------
// One arena per worker slot (slot 0 = whole-decode / inner-16-thread; >0 = chunk-parallel).
struct Arena{
    int cap=0;                                  // token capacity (max M)
    std::vector<float> x,xt,h,qkv,ao,glu;       // f32 activations
    std::vector<bf16>  srcb;                     // bf16 GEMM src staging (max K=2304)
    void ensure(int M){
        if(M<=cap) return; cap=M;
        x.assign((size_t)M*DIM,0); xt.assign((size_t)M*DIM,0); h.assign((size_t)M*DIM,0);
        qkv.assign((size_t)M*QKV,0); ao.assign((size_t)M*DIM,0); glu.assign((size_t)M*GLU2,0);
        srcb.assign((size_t)M*GLU2,0);            // big enough for any src (K<=2304) and glu val stage
    }
};
static std::vector<Arena> AR;

// ------------------------- fp32 elementwise kernels (OMP over slot's own thread budget) -------------------------
// par=true: kernels use `#pragma omp parallel for` (slot 0, all threads). par=false: run serial
// (already inside an outer chunk-parallel region -> no nested fork; oneDNN also serializes there).

// DyT: out[m,j] = gamma[j]*tanh(alpha*x[m,j]) + beta[j]  (full-DIM norm, gk==K)
static void dyt(const float*x,float*o,int M,int K,float alpha,const float*g,const float*b,int gk,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        const float* xr=x+(size_t)m*K; float* orr=o+(size_t)m*K;
        #pragma omp simd
        for(int j=0;j<K;j++) orr[j]=g[j]*vtanh(alpha*xr[j])+b[j];
    }
}
// DyT on a DIM-wide slice of a wide row (q/k/qd/kd of qkv[M,QKV]); head-tiled gamma[64].
// Loop per head so the inner-64 is contiguous & vectorizable (gamma index = d, not j%64).
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
// RoPE in place on a DIM-wide slice (12 heads x 64), per-token position p=(m%ECH)
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
// add bias (broadcast [N]) into c[M,N]
static void addbias(float*c,const float*bias,int M,int N,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){ float* r=c+(size_t)m*N; for(int j=0;j<N;j++) r[j]+=bias[j]; }
}
// residual add: x += y  (M*DIM)
static void resadd(float*x,const float*y,int M,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(size_t i=0;i<(size_t)M*DIM;i++) x[i]+=y[i];
}
// stage f32 rows -> bf16 (M*K)
static void to_bf16(const float*x,bf16*o,size_t n,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(size_t i=0;i<n;i++) o[i]=f2b(x[i]);
}

// fp32 differential attention over the nc 34-token chunks, H heads. Reads q/k/v/qd/kd from
// qkv[M,QKV] (col blocks 0,768,1536,2304,3072). Writes out[M,DIM]. out = SM(q k^T s)v - SM(qd kd^T s)v.
static void diff_attn(const float*qkv,int M,float* out,bool par){
    int nc=M/ECH;
    #pragma omp parallel for collapse(2) schedule(static) if(par)
    for(int c=0;c<nc;c++)for(int hh=0;hh<H;hh++){
        const int base=c*ECH;
        // gather this head's 5 slices into contiguous [34,64] L1 buffers ONCE (kill the
        // 34x redundant L2 re-reads of k/v across query rows; QK/PV then dense & SIMD-clean)
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
// slot: arena/matmul cache index (0 = 16-thread; >0 = a chunk-parallel worker running serial).
// par:  true -> kernels use omp-for (slot 0); false -> serial (inside outer chunk-parallel).
static void decode(int slot,bool par,const float* latent,int T,float* out_patches){
    Arena& A=AR[slot];
    int iT=SUB*T;              // internal tokens after new-token expansion (17T)
    int M2=iT+ECH;             // padded token count for the shifted 2nd half
    A.ensure(M2);
    float* x=A.x.data(); float* xt=A.xt.data(); float* h=A.h.data();
    float* qkv=A.qkv.data(); float* ao=A.ao.data(); float* glu=A.glu.data();
    bf16*  srcb=A.srcb.data();
    float rstd=F32("running_std")[0];

    // project_in: build src [T,256] bf16 = (latent^T * running_std) rows, then GEMM -> [T,768]
    // latent is [1,256,T] channel-major: latent[c*T + t]
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){ bf16* r=srcb+(size_t)t*LAT; for(int c=0;c<LAT;c++) r[c]=f2b(latent[(size_t)c*T+t]*rstd); }
    gemm_bf16(slot,srcb,BF("project_in.w"),T,DIM,LAT,xt);   // xt[T,768]
    addbias(xt,F32("project_in.b"),T,DIM,par);
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
        // ---- attention: h = pre_norm(x); qkv = to_qkv(h); dyt+rope; attn; ao = to_out(.); x += ao
        dyt(x,h,M,DIM,F32(W("pre.alpha"))[0],F32(W("pre.gamma")),F32(W("pre.beta")),DIM,par); TB(2);
        to_bf16(h,srcb,(size_t)M*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("qkv.w")),M,QKV,DIM,qkv); TB(0);
        // q_norm on q(0) & qd(3); k_norm on k(1) & kd(4); (v at 2 untouched)
        dyt_slice(qkv,M,QKV,0*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,M,QKV,3*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,M,QKV,1*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par);
        dyt_slice(qkv,M,QKV,4*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par); TB(2);
        rope_slice(qkv,M,QKV,0*DIM,par); rope_slice(qkv,M,QKV,1*DIM,par);
        rope_slice(qkv,M,QKV,3*DIM,par); rope_slice(qkv,M,QKV,4*DIM,par); TB(3);
        diff_attn(qkv,M,ao,par); TB(1);   // ao[M,DIM] = attention output (fp32)
        to_bf16(ao,srcb,(size_t)M*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("out.w")),M,DIM,DIM,h); TB(0);    // h = to_out(ao)
        resadd(x,h,M,par); TB(6);
        // ---- FFN: h = ff_norm(x); glu = glu_proj(h); val*silu(gate); proj_out; x += .
        dyt(x,h,M,DIM,F32(W("ff.alpha"))[0],F32(W("ff.gamma")),F32(W("ff.beta")),DIM,par); TB(2);
        to_bf16(h,srcb,(size_t)M*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("glu.w")),M,GLU2,DIM,glu); TB(0);
        addbias(glu,F32(W("glu.b")),M,GLU2,par); TB(6);
        // fuse GLU-SiLU + bf16 requant: h_ff = value*silu(gate) -> bf16 staging for proj GEMM
        #pragma omp parallel for schedule(static) if(par)
        for(int m=0;m<M;m++){
            const float* v=glu+(size_t)m*GLU2; const float* g=v+FF; bf16* r=srcb+(size_t)m*FF;
            #pragma omp simd
            for(int j=0;j<FF;j++) r[j]=f2b(v[j]*vsilu(g[j]));
        } TB(4);
        gemm_bf16(slot,srcb,BF(W("proj.w")),M,DIM,FF,h); TB(0);    // h = proj_out(.)
        addbias(h,F32(W("proj.b")),M,DIM,par); resadd(x,h,M,par); TB(6);
    };

    // blocks 0..2 over the iT tokens (nc1 = iT/34 chunks)
    for(int b=0;b<3;b++) run_block(b,iT);
    // midpoint shift by 17: build padded stream [x[0:17], x, x[iT-17:iT]] (len iT+34) in xt,
    // then copy back into x so the block loop keeps operating on A.x.
    std::memcpy(xt, x, (size_t)SHIFT*DIM*sizeof(float));
    std::memcpy(xt+(size_t)SHIFT*DIM, x, (size_t)iT*DIM*sizeof(float));
    std::memcpy(xt+(size_t)(SHIFT+iT)*DIM, x+(size_t)(iT-SHIFT)*DIM, (size_t)SHIFT*DIM*sizeof(float));
    std::memcpy(x, xt, (size_t)M2*DIM*sizeof(float));
    // blocks 3..5 over M2 tokens (nc2 = M2/34)
    for(int b=3;b<6;b++) run_block(b,M2);
    // strip shift: keep x[17 : 17+iT]
    // drop latent slot: y[t*16 + (s-1)] = x[SHIFT + t*17 + s], s=1..16  -> [16T, DIM]
    int L=SIN*T;
    float* y=A.h.data();      // reuse h as [16T,DIM] staging (16T*768 <= M2*768)
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        for(int s=1;s<SUB;s++)
            std::memcpy(y+(size_t)(t*SIN+(s-1))*DIM, x+(size_t)(SHIFT+t*SUB+s)*DIM, DIM*sizeof(float));
    }
    // conv1d 768->512 k3 pad1 via im2col: cols[L, 2304] bf16 (col index c*3+k), GEMM -> [L,512], +bias, transpose
    bf16* cols=srcb;          // [L, 2304] bf16  (L*2304 <= M2*GLU2)
    #pragma omp parallel for schedule(static) if(par)
    for(int l=0;l<L;l++){
        bf16* r=cols+(size_t)l*(DIM*3);
        for(int c=0;c<DIM;c++)for(int k=0;k<3;k++){
            int j=l+k-1;                       // pad=1
            float v=(j>=0&&j<L)? y[(size_t)j*DIM+c] : 0.0f;
            r[c*3+k]=f2b(v);
        }
    }
    float* cy=A.qkv.data();   // [L,512] f32  (L*512 <= M2*QKV)
    gemm_bf16(slot,cols,BF("conv.w"),L,OUTCH,DIM*3,cy);
    addbias(cy,F32("conv.b"),L,OUTCH,par);
    // transpose [L,512] -> patches[512,L]  (out_patches = [1,512,L], ch-major)
    #pragma omp parallel for schedule(static) if(par)
    for(int ch=0;ch<OUTCH;ch++)for(int l=0;l<L;l++) out_patches[(size_t)ch*L+l]=cy[(size_t)l*OUTCH+ch];
}

// ============================== C ABI ==============================
static int NTHREADS=16;
extern "C" {

int sames_init(const char* weights_base,int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    NTHREADS=threads;
    if(weights_base && weights_base[0]) WBASE=weights_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[sames] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    omp_set_max_active_levels(1);   // chunk-parallel is level 1; oneDNN inside serializes
    #pragma omp parallel num_threads(threads)
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init(); mmcache_init(threads);   // slot 0 (16-thread) + 1..threads chunk-parallel workers
    load_weights(); build_rope();
    AR.resize(threads+1);
    PROF_ON = getenv("SAMES_PROF")!=nullptr;
    printf("[sames] init ok: threads=%d isa=%s\n",threads,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));
    fflush(stdout);
    return 0;
}

// whole decode: latent [1,256,T] f32 -> out_patches [1,512,16T] f32 (caller-allocated)
void sames_forward(const float* latent,int T,float* out_patches){
    decode(0,true,latent,T,out_patches);
}

// torch-free unpatch: patches[1,512,L] -> pcm[1,2,256L]. reshape[2,256,L]->transpose->[2,L,256]->[2,256L]
void sames_unpatch(const float* patches,int L,float* pcm){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int st=0;st<2;st++)for(int l=0;l<L;l++){
        float* o=pcm+((size_t)st*L+l)*256;
        for(int p=0;p<256;p++) o[p]=patches[((size_t)st*256+p)*L+l];
    }
}

int sames_DIM(){return DIM;} int sames_SUB(){return SUB;} int sames_SIN(){return SIN;}

// print + reset accumulated phase times (ms total across calls since last dump)
void sames_prof_dump(){
    double tot=0; for(int i=0;i<7;i++) tot+=PROF[i];
    printf("[prof] ");
    for(int i=0;i<7;i++) printf("%s=%.1fms(%.0f%%) ",PROFLBL[i],PROF[i]*1e3,tot>0?100*PROF[i]/tot:0);
    printf(" total=%.1fms\n",tot*1e3);
    for(int i=0;i<8;i++) PROF[i]=0; PROFN=0; fflush(stdout);
}

} // extern "C"

// ---- chunk-parallel decode is appended in a second translation section below via include ----
#include "same_s_chunk.inc"
