// same_l_cpu_amx.cpp — torch-free C++ AMX-BF16 SAME-L decoder as a callable .so.
//
// SAME-L = native 426M medium decoder (12 blocks, dim=1536, 24 heads). Mirrors the proven
// SAME-S engine (same_s_cpu_amx.cpp) but with the SAME-L architecture differences:
//   * BANDED SWA differential attention (window +-17 over the internal 17*T sequence),
//     implemented directly in fp32 as a LINEAR band scan (no dense [S,S] mask -> no O(T^2)).
//   * GLOBAL RoPE positions over the internal sequence (relative RoPE => chunk-local == global).
//   * sin-gate FF for blocks 5..11 (value*sin(pi*gate)); silu for blocks 0..4.
//   * plain Linear(1536->512) output map (no WNConv1d / im2col).
//   * NO midpoint-shift (that is a SAME-S 34-token-chunk trick).
// oneDNN AMX-BF16 GEMMs for every linear; fp32 C++ for the cancellation-fragile elementwise
// (DyT norms, RoPE, GLU) and the differential band attention.
//
//   samel_init(weights_base, threads)                              -> mmap bf16 weights + AMX/omp/oneDNN
//   samel_forward(latent[1,256,T], T, out_patches[1,512,16T])      -> whole decode (linear band attn)
//   samel_forward_chunked(latent, T, C, overlap, parallel, out)    -> chunked decode (PRIMARY: C=64,ovl=8)
//   samel_unpatch(patches[1,512,L], L, pcm[1,2,256L])              -> torch-free unpatch
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            same_l_cpu_amx.cpp -o same_l_cpu_amx.so $ONELIB/libdnnl.a -ldl -lpthread -lm
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
static const int SUB=17, SIN=16;               // 17 internal tok/latent, keep 16 (drop slot 0)
static const int SIN_START=5;                  // blocks >=5 use sin(pi*gate) gate
static const int BAND=17;                       // SWA half-window (BLOCK_SIZE=SUB_CHUNK_SIZE=17)
static const float SCALE=0.125f;                // HD**-0.5 = 64**-0.5

// ── optional phase profiler (SAMEL_PROF=1) ──
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
// sin(pi*g): range-reduce g to r in [-0.5,0.5] (g=k+r), sin(pi*g)=(-1)^k * sin(pi*r);
// pi*r in [-pi/2,pi/2] via degree-9 Taylor (err ~2e-6 << bf16). branchless -> SIMD-clean.
static inline float vsinpi(float g){
    float k = std::floor(g+0.5f);
    float r = g - k;                              // [-0.5,0.5]
    float s = 1.0f - 2.0f*(float)(((long)k)&1L);  // (-1)^k
    float y = 3.14159265358979f*r;                // [-pi/2,pi/2]
    float y2 = y*y;
    float p = y*(1.0f + y2*(-0.16666667f + y2*(0.00833333f + y2*(-0.00019841f + y2*2.75573e-6f))));
    return s*p;
}
static inline int cdiv(int a,int b){return (a+b-1)/b;}

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
static std::string WBASE = std::string(sa3_home()) + "/same_l_cpu_amx/weights";
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
    printf("[samel] weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
}
static float* F32(const std::string&k){return (float*)TEN.at(k).p;}
static bf16*  BF (const std::string&k){return (bf16*)TEN.at(k).p;}

// ------------------------- oneDNN bf16 matmul (bf16 x bf16 -> f32): primitive+handle cache
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
// src A[M,K] bf16, wei B[K,N] bf16 (row-major), dst C[M,N] f32. cache slot `slot`.
static void gemm_bf16(int slot,const bf16*A,const bf16*B,int M,int N,int K,float* C){
    using dt=dnnl::memory::data_type;
    MMCache& c=MMC[slot];
    MMKey key{M,N,K}; auto it=c.mm.find(key);
    if(it==c.mm.end()){
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

// ------------------------- GLOBAL RoPE table (positions 0..cap-1, 16 freqs) -------------------------
// SAME-L RoPE runs over the whole internal sequence. Relative RoPE => a chunk's local positions
// give identical band-attention scores as global positions, so decode() uses absolute index m.
static float RINV[HALF];
static std::vector<float> RCOS, RSIN;  // [cap*HALF]
static int RCAP=0;
static void rope_invfreq(){ for(int i=0;i<HALF;i++) RINV[i]=std::pow(10000.0f,-(float)(2*i)/(float)RD); }
static void rope_fill(int lo,int hi){
    for(int p=lo;p<hi;p++) for(int i=0;i<HALF;i++){
        RCOS[(size_t)p*HALF+i]=std::cos(p*RINV[i]);
        RSIN[(size_t)p*HALF+i]=std::sin(p*RINV[i]);
    }
}
static void rope_reserve(int cap){                 // called at init (single-thread) to a safe max
    if(cap<=RCAP) return;
    RCOS.resize((size_t)cap*HALF); RSIN.resize((size_t)cap*HALF);
    rope_fill(RCAP,cap); RCAP=cap;
}
static void ensure_rope(int S){                    // grow if a (serial) whole-decode needs more
    if(S<=RCAP) return;
    #pragma omp critical(rope_grow)
    { if(S>RCAP){ int nc=S+S/4; RCOS.resize((size_t)nc*HALF); RSIN.resize((size_t)nc*HALF);
                  rope_fill(RCAP,nc); RCAP=nc; } }
}

// ------------------------- reusable arena (per worker slot) -------------------------
struct Arena{
    int cap=0;
    std::vector<float> x,xt,h,qkv,ao,glu;     // f32 activations
    std::vector<bf16>  srcb;                    // bf16 GEMM src staging
    void ensure(int M){
        if(M<=cap) return; cap=M;
        x.assign((size_t)M*DIM,0); xt.assign((size_t)M*DIM,0); h.assign((size_t)M*DIM,0);
        qkv.assign((size_t)M*QKV,0); ao.assign((size_t)M*DIM,0); glu.assign((size_t)M*GLU2,0);
        srcb.assign((size_t)M*GLU2,0);          // wide enough for any src (K<=FF=4608) and glu val stage
    }
};
static std::vector<Arena> AR;

// ------------------------- fp32 elementwise kernels -------------------------
// DyT: out[m,j] = gamma[j]*tanh(alpha*x[m,j]) + beta[j]
static void dyt(const float*x,float*o,int M,int K,float alpha,const float*g,const float*b,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){
        const float* xr=x+(size_t)m*K; float* orr=o+(size_t)m*K;
        #pragma omp simd
        for(int j=0;j<K;j++) orr[j]=g[j]*vtanh(alpha*xr[j])+b[j];
    }
}
// DyT on a DIM-wide slice of the wide qkv row (q1/k1/q2/k2); head-tiled gamma[64].
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
// RoPE in place on a DIM-wide slice (24 heads x 64), GLOBAL position p=m
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
static void addbias(float*c,const float*bias,int M,int N,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int m=0;m<M;m++){ float* r=c+(size_t)m*N; for(int j=0;j<N;j++) r[j]+=bias[j]; }
}
static void resadd(float*x,const float*y,int M,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(size_t i=0;i<(size_t)M*DIM;i++) x[i]+=y[i];
}
static void to_bf16(const float*x,bf16*o,size_t n,bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(size_t i=0;i<n;i++) o[i]=f2b(x[i]);
}

// ------------------------- BANDED differential attention (fp32) -------------------------
// out[i] = SM_band(q1 k1^T s) v - SM_band(q2 k2^T s) v, band |i-j|<=17 over the internal S tokens.
// q1,k1,v,q2,k2 live in qkv[S,QKV] col blocks 0,1,2,3,4 (each DIM wide). Parallel over heads;
// each head gathers its 5 slices into contiguous [S,HD] buffers ONCE (kills the QKV=7680 stride).
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
        const int WN=2*BAND+1;                    // <=35 keys per query
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
            (void)WN;
        }
    }
}

// ------------------------- one whole decode (latent[256,T] -> patches[512,16T]) ---------
// slot: arena/matmul cache index. par: true -> kernels use omp-for (slot 0); false -> serial.
static void decode(int slot,bool par,const float* latent,int T,float* out_patches){
    Arena& A=AR[slot];
    int S=SUB*T;                    // internal tokens after new-token expansion (17T)
    A.ensure(S);
    ensure_rope(S);
    float* x=A.x.data(); float* xt=A.xt.data(); float* h=A.h.data();
    float* qkv=A.qkv.data(); float* ao=A.ao.data(); float* glu=A.glu.data();
    bf16*  srcb=A.srcb.data();
    float rstd=F32("running_std")[0];

    // project_in: src [T,256] bf16 = (latent^T * running_std) rows, GEMM -> [T,1536], +bias
    // latent is [1,256,T] channel-major: latent[c*T + t]
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){ bf16* r=srcb+(size_t)t*LAT; for(int c=0;c<LAT;c++) r[c]=f2b(latent[(size_t)c*T+t]*rstd); }
    gemm_bf16(slot,srcb,BF("project_in.w"),T,DIM,LAT,xt);   // xt[T,1536]
    addbias(xt,F32("project_in.b"),T,DIM,par);
    // expand: token t*17+0 = xt[t]; t*17+1..16 = new_tokens (single learned vector)
    const float* ntk=F32("new_tokens");
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++){
        std::memcpy(x+(size_t)(t*SUB)*DIM, xt+(size_t)t*DIM, DIM*sizeof(float));
        for(int s=1;s<SUB;s++) std::memcpy(x+(size_t)(t*SUB+s)*DIM, ntk, DIM*sizeof(float));
    }

    auto run_block=[&](int b){
        double _pt=wt();
        bool use_sin=(b>=SIN_START);
        char pb[8]; snprintf(pb,sizeof pb,"b%d.",b);
        auto W=[&](const char*n){return std::string(pb)+n;};
        // ---- attention: h = pre_norm(x); qkv = to_qkv(h); dyt+rope; band-attn; ao=to_out; x += ao
        dyt(x,h,S,DIM,F32(W("pre.alpha"))[0],F32(W("pre.gamma")),F32(W("pre.beta")),par); TB(2);
        to_bf16(h,srcb,(size_t)S*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("qkv.w")),S,QKV,DIM,qkv); TB(0);
        dyt_slice(qkv,S,QKV,0*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,S,QKV,3*DIM,F32(W("qn.alpha"))[0],F32(W("qn.gamma")),F32(W("qn.beta")),par);
        dyt_slice(qkv,S,QKV,1*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par);
        dyt_slice(qkv,S,QKV,4*DIM,F32(W("kn.alpha"))[0],F32(W("kn.gamma")),F32(W("kn.beta")),par); TB(2);
        rope_slice(qkv,S,QKV,0*DIM,par); rope_slice(qkv,S,QKV,1*DIM,par);
        rope_slice(qkv,S,QKV,3*DIM,par); rope_slice(qkv,S,QKV,4*DIM,par); TB(3);
        diff_attn_banded(qkv,S,ao,par); TB(1);
        to_bf16(ao,srcb,(size_t)S*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("out.w")),S,DIM,DIM,h); TB(0);
        resadd(x,h,S,par); TB(6);
        // ---- FFN: h = ff_norm(x); glu = glu_proj(h); val*act(gate); proj_out; x += .
        dyt(x,h,S,DIM,F32(W("ff.alpha"))[0],F32(W("ff.gamma")),F32(W("ff.beta")),par); TB(2);
        to_bf16(h,srcb,(size_t)S*DIM,par); TB(5);
        gemm_bf16(slot,srcb,BF(W("glu.w")),S,GLU2,DIM,glu); TB(0);
        addbias(glu,F32(W("glu.b")),S,GLU2,par); TB(6);
        // fuse GLU gate + bf16 requant: h_ff = value*act(gate) -> bf16 for proj GEMM
        // blocks 0..4: act=silu;  blocks 5..11: act=sin(pi*gate)
        #pragma omp parallel for schedule(static) if(par)
        for(int m=0;m<S;m++){
            const float* v=glu+(size_t)m*GLU2; const float* g=v+FF; bf16* r=srcb+(size_t)m*FF;
            if(use_sin){ for(int j=0;j<FF;j++) r[j]=f2b(v[j]*vsinpi(g[j])); }
            else       { for(int j=0;j<FF;j++) r[j]=f2b(v[j]*vsilu(g[j])); }
        } TB(4);
        gemm_bf16(slot,srcb,BF(W("proj.w")),S,DIM,FF,h); TB(0);
        addbias(h,F32(W("proj.b")),S,DIM,par); resadd(x,h,S,par); TB(6);
    };

    for(int b=0;b<NB;b++) run_block(b);

    // drop latent slot 0 of each 17-block: y[t*16 + (s-1)] = x[t*17 + s], s=1..16  -> [16T, DIM]
    int L=SIN*T;
    float* y=A.h.data();          // reuse h as [16T,DIM] staging (16T*1536 <= S*1536)
    #pragma omp parallel for schedule(static) if(par)
    for(int t=0;t<T;t++)
        for(int s=1;s<SUB;s++)
            std::memcpy(y+(size_t)(t*SIN+(s-1))*DIM, x+(size_t)(t*SUB+s)*DIM, DIM*sizeof(float));
    // output map: plain Linear(1536->512). y[L,1536] bf16 -> GEMM -> [L,512] f32, +bias, transpose
    to_bf16(y,srcb,(size_t)L*DIM,par);
    float* cy=A.qkv.data();       // [L,512] f32  (L*512 <= S*QKV)
    gemm_bf16(slot,srcb,BF("map.w"),L,OUTCH,DIM,cy);
    addbias(cy,F32("map.b"),L,OUTCH,par);
    #pragma omp parallel for schedule(static) if(par)
    for(int ch=0;ch<OUTCH;ch++)for(int l=0;l<L;l++) out_patches[(size_t)ch*L+l]=cy[(size_t)l*OUTCH+ch];
}

// ============================== C ABI ==============================
static int NTHREADS=16;
extern "C" {

int samel_init(const char* weights_base,int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    NTHREADS=threads;
    if(weights_base && weights_base[0]) WBASE=weights_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[samel] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    omp_set_max_active_levels(1);
    #pragma omp parallel num_threads(threads)
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init(); mmcache_init(threads);
    load_weights(); rope_invfreq();
    rope_reserve(SUB*1024);        // covers all chunks + whole-decode T<=1024 without a realloc race
    AR.resize(threads+1);
    PROF_ON = getenv("SAMEL_PROF")!=nullptr;
    printf("[samel] init ok: threads=%d isa=%s\n",threads,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));
    fflush(stdout);
    return 0;
}

// whole decode: latent [1,256,T] f32 -> out_patches [1,512,16T] f32 (caller-allocated)
void samel_forward(const float* latent,int T,float* out_patches){
    decode(0,true,latent,T,out_patches);
}

// torch-free unpatch: patches[1,512,L] -> pcm[1,2,256L]. reshape[2,256,L]->transpose->[2,L,256]->[2,256L]
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
#include "same_l_chunk.inc"
