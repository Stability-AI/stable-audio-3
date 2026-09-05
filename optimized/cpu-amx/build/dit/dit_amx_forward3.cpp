// Speedprove v3: optimized TORCH-FREE C++ driver for the SA3-medium int8 DiT.
// Same math/ABI as dit_amx_forward2.cpp (bit-exact), but the orchestration layer is rewritten:
//   1. Preallocated, reused scratch pools (NO per-call malloc, NO zero-init of overwritten buffers).
//   2. oneDNN dst-buffer + dnnl::memory-handle reuse (set_data_handle) alongside the primitive cache.
//   3. OMP-parallelized slice_to_heads/from_heads (now memcpy), ssg add, no post xt copy.
// Bit-exact vs the model_p3(int8/tri) golden (int8@int8->int32 is integer-exact). Timing harness @L.
//
// build: see RESULTS_speedprove.md
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <omp.h>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

static const int E=1536, H=24, D=64, RD=32, MEM=64, CROSS=257, DEPTH=24;
static const int FF=6144;                       // FFN inner
static const float NORM_EPS=1e-5f, QK_EPS=1e-6f;
static const float SCALE=0.125f;                // D**-0.5 = 64**-0.5

// Engine paths resolve from $SA3_CPUAMX_HOME (same base the Python side uses), so nothing
// absolute is baked into the binary. Falls back to the current directory.
static const char* sa3_home() {
    const char* v = getenv("SA3_CPUAMX_HOME");
    return (v && *v) ? v : ".";
}
static std::string AOT_S = std::string(sa3_home()) + "/dit_medium_cpu_amx/aot_stage2";
static const char* AOT = AOT_S.c_str();        // so/ + cpp_kernels.txt (L-independent)
static std::string COREBASE = std::string(sa3_home()) + "/dit_medium_cpu_amx/core_L128";  // .bin + _manifest.txt

static bool USE_ONEDNN=true;                     // int8 linear GEMM backend
static void* FLASH_OVERRIDE=nullptr;             // optional alternate flash kernel .so entry
static int   FLASH_BM=128;                       // flash query-block size (BM); constexpr in the .so.
// BM only tiles the (independent) query rows -> bit-exact vs BM=32, but 4x fewer per-tile kernel
// invocations, cutting the per-launch AMX-config/dispatch overhead. bm{64,128}.so live in so_flash/.
static std::string FLASH_SO_DIR_S = std::string(sa3_home()) + "/dit_medium_cpu_amx/so_flash";
static const char* FLASH_SO_DIR = FLASH_SO_DIR_S.c_str();

static inline int cdiv(int a,int b){return (a+b-1)/b;}
static int npow2(int n){int p=1;while(p<n)p<<=1;return p;}
typedef unsigned u32;

// ------------------------- mmap core.bin + manifest -------------------------
struct Ten{void* p; std::string dt; long n; std::vector<long> shp;};
static std::map<std::string,Ten> TEN;
static char* BASE=nullptr;
static void load_core(){
    std::string bin=COREBASE+".bin";
    int fd=open(bin.c_str(),O_RDONLY); struct stat st; fstat(fd,&st);
    BASE=(char*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(BASE==MAP_FAILED){perror("mmap");exit(1);} close(fd);
    std::ifstream mf(COREBASE+"_manifest.txt"); std::string line;
    while(std::getline(mf,line)){
        std::istringstream ss(line); Ten t; std::string name; long off;
        ss>>name>>t.dt>>off>>t.n; long d; while(ss>>d)t.shp.push_back(d);
        t.p=(void*)(BASE+off); TEN[name]=t;
    }
    printf("core.bin mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
}
static float*  F32(const std::string&k){return (float*)TEN.at(k).p;}
static int8_t* I8 (const std::string&k){return (int8_t*)TEN.at(k).p;}

// ------------------------- dlopen dispatch table -------------------------
static std::map<std::string,void*> FN;
static void load_kernels(){
    std::ifstream kf(std::string(AOT)+"/cpp_kernels.txt"); std::string key,so,sym;
    while(kf>>key>>so>>sym){
        std::string path=std::string(AOT)+"/so/"+so;
        void* h=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!h){fprintf(stderr,"dlopen %s: %s\n",path.c_str(),dlerror());exit(1);}
        void* f=dlsym(h,sym.c_str());
        if(!f){fprintf(stderr,"dlsym %s\n",sym.c_str());exit(1);}
        FN[key]=f;
    }
    printf("dlopen'd %ld AOT kernels\n",(long)FN.size());
}

// ------------------------- kernel ABI typedefs (surviving args + 6 grid u32) -------------------------
typedef void(*t_gemm_i8)(void*,void*,void*,int,int,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_gemm_fp)(void*,void*,void*,void*,int,int,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_quant)(void*,void*,void*,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_rms)(void*,void*,void*,int,int,int,int,float,u32,u32,u32,u32,u32,u32);
typedef void(*t_rope)(void*,void*,void*,void*,int,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_flash)(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
                       int,int,float,int,int,int,int,int,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_rmsmodq)(void*,void*,void*,void*,void*,void*,int,int,int,int,float,u32,u32,u32,u32,u32,u32);
typedef void(*t_deq)(void*,void*,void*,void*,void*,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_deqglu)(void*,void*,void*,void*,void*,void*,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_deqgate)(void*,void*,void*,void*,void*,void*,void*,int,int,int,int,int,u32,u32,u32,u32,u32,u32);
typedef void(*t_deqadd)(void*,void*,void*,void*,void*,void*,int,int,int,int,int,int,u32,u32,u32,u32,u32,u32);

// ------------------------- reusable scratch pools (allocated once, no zero-init) -------------------------
template<class T> struct Pool{
    std::vector<std::vector<T>> slab; size_t elems=0; int cur=0;
    void init(int n,size_t e){ elems=e; slab.assign(n,std::vector<T>(e)); cur=0; }
    T* get(){ if(cur>=(int)slab.size()){fprintf(stderr,"POOL OVERFLOW (%d slabs)\n",(int)slab.size());exit(3);} return slab[cur++].data(); }
    void reset(){ cur=0; }
};
static Pool<float>   HEADf;   // [H,S,D]=S*E sized head-space tensors
static Pool<float>   WIDEf;   // deq outputs: qkv[S,5E], q2[S,2E], kv[CROSS,3E], proj/post
static Pool<int8_t>  QROW;    // int8 row-quant activations up to [S,FF]
static Pool<float>   SROWf;   // row-scale vectors [S]/[Mp]
static std::vector<int32_t> ACC;        // single shared GEMM int32 accumulator (max S*2FF)
static std::vector<float>   XA,XB,RES,SSG;   // x ping-pong, residual snapshot, ssg buffer
// flash internal scratch (reused every flash call; calls are sequential)
static std::vector<int8_t>  FL_qmi,FL_kmi,FL_qdi,FL_kdi,FL_viq;
static std::vector<float>   FL_out,FL_qsm,FL_ksm,FL_qsd,FL_ksd,FL_vsc;

static void alloc_scratch(int S){
    // Head-space & flash tensors span S rows (self-attn) OR CROSS rows (cross-attn K/V side);
    // size them to max(S,CROSS) so short sequences (S<CROSS, e.g. L128) don't overflow.
    int Sm=std::max(S,CROSS); size_t se=(size_t)S*E, sme=(size_t)Sm*E, hsm=(size_t)H*Sm;
    HEADf.init(16, sme);
    WIDEf.init(4,  (size_t)S*5*E);               // qkv[S,5E] >= kv[CROSS,3E] for all S>=... (holds L128 too)
    QROW .init(8,  (size_t)S*FF);
    SROWf.init(8,  (size_t)Sm);
    ACC.assign((size_t)S*2*FF,0);
    XA.assign(se,0); XB.assign(se,0); RES.assign(se,0); SSG.assign((size_t)6*E,0);
    FL_qmi.assign(sme,0);FL_kmi.assign(sme,0);FL_qdi.assign(sme,0);FL_kdi.assign(sme,0);FL_viq.assign(sme,0);
    FL_out.assign(sme,0);FL_qsm.assign(hsm,0);FL_ksm.assign(hsm,0);FL_qsd.assign(hsm,0);FL_ksd.assign(hsm,0);FL_vsc.assign(hsm,0);
    printf("scratch: HEADf16x%.1fMB WIDEf4x%.1fMB QROW8x%.1fMB ACC%.1fMB (S=%d Sm=%d)\n",
           sme*4/1e6,(double)S*5*E*4/1e6,(double)S*FF/1e6,(double)S*2*FF*4/1e6,S,Sm);
}
static void reset_block_pools(){ HEADf.reset(); WIDEf.reset(); QROW.reset(); SROWf.reset(); }

// ------------------------- oneDNN int8 GEMM (s8 x s8 -> s32): primitive + memory-handle cache ---------
static dnnl::engine* ENG=nullptr;
static dnnl::stream* STRM=nullptr;
struct MMKey{int M,N,K; bool operator<(const MMKey&o)const{
    return M!=o.M?M<o.M:(N!=o.N?N<o.N:K<o.K);}};
struct MMEnt{dnnl::matmul prim; dnnl::memory am,bm,cm;};
static std::map<MMKey,MMEnt> MM;
static void onednn_init(){
    ENG=new dnnl::engine(dnnl::engine::kind::cpu,0);
    STRM=new dnnl::stream(*ENG);
}
// write s8[M,K] x s8[K,N] -> s32[M,N] into caller-provided dst (no alloc, no zero-init)
static void gemm_i8_onednn(const int8_t*a,const int8_t*b,int M,int N,int K,int32_t* dst){
    using dt=dnnl::memory::data_type;
    MMKey key{M,N,K}; auto it=MM.find(key);
    if(it==MM.end()){
        dnnl::memory::desc a_md({M,K},dt::s8, {K,1});   // A row-major [M,K]
        dnnl::memory::desc b_md({K,N},dt::s8, {N,1});   // B row-major [K,N] (plain, matches _int_mm)
        dnnl::memory::desc c_md({M,N},dt::s32,{N,1});   // C row-major [M,N]
        dnnl::matmul::primitive_desc pd(*ENG,a_md,b_md,c_md);
        MMEnt e{dnnl::matmul(pd),
                dnnl::memory(pd.src_desc(),*ENG,(void*)a),
                dnnl::memory(pd.weights_desc(),*ENG,(void*)b),
                dnnl::memory(pd.dst_desc(),*ENG,(void*)dst)};
        it=MM.emplace(key,std::move(e)).first;
    }
    MMEnt& e=it->second;
    e.am.set_data_handle((void*)a);
    e.bm.set_data_handle((void*)b);
    e.cm.set_data_handle((void*)dst);
    e.prim.execute(*STRM,{{DNNL_ARG_SRC,e.am},{DNNL_ARG_WEIGHTS,e.bm},{DNNL_ARG_DST,e.cm}});
    STRM->wait();
}

// ------------------------- kernel wrappers (grid looped in C, OMP-parallel; write into dst) ----------
static void gemm_i8_triton(const int8_t*a,const int8_t*b,int M,int N,int K,int32_t* dst){
    std::memset(dst,0,(size_t)M*N*sizeof(int32_t)); auto fn=(t_gemm_i8)FN.at("gemm_i8"); // triton path accumulates
    u32 g=cdiv(M,32)*cdiv(N,64);
    #pragma omp parallel for schedule(static)
    for(u32 x=0;x<g;x++) fn((void*)a,(void*)b,dst,M,N,K,K,N,N,x,0,0,g,1,1);
}
static inline void gemm_i8(const int8_t*a,const int8_t*b,int M,int N,int K,int32_t* dst){
    if(USE_ONEDNN) gemm_i8_onednn(a,b,M,N,K,dst); else gemm_i8_triton(a,b,M,N,K,dst);
}
static void gemm_fp(const float*a,const float*b,int M,int N,int K,float* dst){
    std::memset(dst,0,(size_t)M*N*sizeof(float)); auto fn=(t_gemm_fp)FN.at("gemm_fp");
    u32 g=cdiv(M,8)*cdiv(N,32);
    #pragma omp parallel for schedule(static)
    for(u32 x=0;x<g;x++) fn((void*)a,(void*)b,dst,(void*)a,M,N,K,K,N,N,x,0,0,g,1,1);
}
static void quant_rows(const float*x,int M,int K,int8_t* q,float* s){
    auto fn=(t_quant)FN.at("quant");
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++) fn((void*)x,q,s,M,K,K,K,m,0,0,M,1,1);
}
static void rmsnorm(const float*x,const float*g,int M,int K,float* o){
    auto fn=(t_rms)FN.at("rmsnorm");
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++) fn((void*)x,(void*)g,o,M,K,K,K,QK_EPS,m,0,0,M,1,1);
}
static void rope(const float*x,const float*cs,const float*sn,int Hh,int S,float* o){
    auto fn=(t_rope)FN.at("rope");
    #pragma omp parallel for collapse(2) schedule(static)
    for(u32 s=0;s<(u32)S;s++)for(u32 h=0;h<(u32)Hh;h++)
        fn((void*)x,(void*)cs,(void*)sn,o,Hh,S,S*D,D,RD,s,h,0,(u32)S,(u32)Hh,1);
}
static void rmsmodq(const char*slot,const float*x,const float*g,const float*sc,const float*sh,
                    int M,int K,int8_t* q,float* s){
    auto fn=(t_rmsmodq)FN.at(slot);
    const float* scp=sc?sc:x; const float* shp=sh?sh:x;
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++)
        fn((void*)x,(void*)g,(void*)scp,(void*)shp,q,s,M,K,K,K,NORM_EPS,m,0,0,M,1,1);
}
static void deq(const int32_t*acc,const float*as,const float*bs,int M,int N,float* o){
    std::string slot="deq_bn"+std::to_string(npow2(N))+"_bias0";
    auto fn=(t_deq)FN.at(slot);
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++) fn((void*)acc,(void*)as,(void*)bs,(void*)as,o,M,N,N,N,m,0,0,M,1,1);
}
static void deqglu(const int32_t*acc,const float*as,const float*bs,const float*bias,
                   int M,int Fdim,int8_t* q,float* s){
    auto fn=(t_deqglu)FN.at("deqglu");
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++)
        fn((void*)acc,(void*)as,(void*)bs,(void*)bias,q,s,M,Fdim,2*Fdim,Fdim,m,0,0,M,1,1);
}
static void deqgate(const int32_t*acc,const float*as,const float*bs,const float*bias,
                    const float*gate,const float*res,int M,int N,float* o){
    std::string slot=bias?"deqgate_bn2048_bias1":"deqgate_bn2048_bias0";
    auto fn=(t_deqgate)FN.at(slot); const float* bp=bias?bias:as;
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++)
        fn((void*)acc,(void*)as,(void*)bs,(void*)bp,(void*)gate,(void*)res,o,M,N,N,N,N,m,0,0,M,1,1);
}
static void deqadd(const int32_t*acc,const float*as,const float*bs,
                   const float*add,const float*local,int M,int N,float* o){
    auto fn=(t_deqadd)FN.at("deqadd_bn2048");
    #pragma omp parallel for schedule(static)
    for(u32 m=0;m<(u32)M;m++)
        fn((void*)acc,(void*)as,(void*)bs,(void*)add,(void*)local,o,M,N,MEM,N,N,N,m,0,0,M,1,1);
}

// int8 differential flash attention: numpy-equivalent per-row input quant, then AMX kernel.
static void quant_hsd(const float*x,int Hh,int S,int8_t* q,float* sc){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<Hh;h++)for(int s=0;s<S;s++){
        const float* r=x+((size_t)h*S+s)*D; float amax=0; for(int d=0;d<D;d++)amax=std::max(amax,std::fabs(r[d]));
        float scale=std::max(amax/127.0f,1e-12f); sc[(size_t)h*S+s]=scale;
        int8_t* qo=q+((size_t)h*S+s)*D;
        for(int d=0;d<D;d++){float v=std::rintf(r[d]/scale); v=std::min(127.0f,std::max(-127.0f,v)); qo[d]=(int8_t)v;}
    }
}
static void flash_i8(const float*qm,const float*km,const float*qd,const float*kd,
                     const float*v,int Sq,int Sk,float* out){
    quant_hsd(qm,H,Sq,FL_qmi.data(),FL_qsm.data()); quant_hsd(km,H,Sk,FL_kmi.data(),FL_ksm.data());
    quant_hsd(qd,H,Sq,FL_qdi.data(),FL_qsd.data()); quant_hsd(kd,H,Sk,FL_kdi.data(),FL_ksd.data());
    int8_t* viq=FL_viq.data(); float* vsc=FL_vsc.data();
    #pragma omp parallel for schedule(static)
    for(int h=0;h<H;h++){float amax=0; for(int i=0;i<Sk*D;i++)amax=std::max(amax,std::fabs(v[(size_t)h*Sk*D+i]));
        float s=std::max(amax/127.0f,1e-12f); vsc[h]=s;
        for(int i=0;i<Sk*D;i++){float q=std::rintf(v[(size_t)h*Sk*D+i]/s);q=std::min(127.0f,std::max(-127.0f,q));viq[(size_t)h*Sk*D+i]=(int8_t)q;}}
    auto fn=FLASH_OVERRIDE?(t_flash)FLASH_OVERRIDE:(t_flash)FN.at("flash");
    u32 gX=cdiv(Sq,FLASH_BM),gY=H;
    #pragma omp parallel for collapse(2) schedule(static)
    for(u32 x=0;x<gX;x++)for(u32 y=0;y<gY;y++)
        fn(FL_qmi.data(),FL_kmi.data(),FL_qdi.data(),FL_kdi.data(),viq,out,
           FL_qsm.data(),FL_ksm.data(),FL_qsd.data(),FL_ksd.data(),vsc,
           Sq,Sk,SCALE,Sq*D,D,Sk*D,D,Sk*D,D,Sq*D,D,H,x,y,0,gX,gY,1);
}

// ------------------------- glue (OMP-parallel; contiguous D blocks -> memcpy) -------------------------
static void slice_to_heads(const float*src,int S,int total,int coloff,float* o){ // [S,total]->[H,S,D]
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++)for(int h=0;h<H;h++)
        std::memcpy(o+((size_t)h*S+s)*D, src+(size_t)s*total+coloff+h*D, D*sizeof(float));
}
static void from_heads(const float*src,int S,float* o){ // [H,S,D]->[S,E]
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++)for(int s=0;s<S;s++)
        std::memcpy(o+(size_t)s*E+h*D, src+((size_t)h*S+s)*D, D*sizeof(float));
}

// ------------------------- one whole forward (24-block int8/AMX core + int8 proj + fp32 post) ------
// returns post [Mp,256] row-major (Mp = S-MEM). Injected golden fp32 preamble (x_init/ctx/gc).
static std::vector<float> run_forward(int S){
    int Mp=S-MEM;
    float* x=XA.data(); float* xnext=XB.data();          // ping-pong for the running activation
    std::memcpy(x,F32("x_init"),(size_t)S*E*sizeof(float));
    int8_t* ctx_i8=I8("ctx_i8"); float* ctx_s=F32("ctx_s"); float* gc=F32("gc");
    float* rc=F32("rope_cos"); float* rs=F32("rope_sin");
    for(int b=0;b<DEPTH;b++){
        reset_block_pools();
        char p[32]; snprintf(p,sizeof p,"b%d.",b);
        auto W=[&](const char*n){return std::string(p)+n;};
        float* ssg=SSG.data();
        {float* sg=F32(W("ssg"));
         #pragma omp parallel for schedule(static)
         for(int i=0;i<6*E;i++) ssg[i]=sg[i]+gc[i];}
        float *sc_s=&ssg[0],*sh_s=&ssg[E],*g_s=&ssg[2*E],*sc_f=&ssg[3*E],*sh_f=&ssg[4*E],*g_f=&ssg[5*E];
        // ===== self-attn =====
        std::memcpy(RES.data(),x,(size_t)S*E*sizeof(float)); float* res=RES.data();
        HEADf.reset();
        int8_t* a_i8=QROW.get(); float* a_s=SROWf.get();
        rmsmodq("rmsmodq_mod1",x,F32(W("pre_g")),sc_s,sh_s,S,E,a_i8,a_s);
        int32_t* acc=ACC.data();
        gemm_i8(a_i8,I8(W("s_qkv.q")),S,5*E,E,acc);
        float* qkv=WIDEf.get(); deq(acc,a_s,F32(W("s_qkv.scale")),S,5*E,qkv);
        float* q =HEADf.get(); slice_to_heads(qkv,S,5*E,0,   q);
        float* k =HEADf.get(); slice_to_heads(qkv,S,5*E,E,   k);
        float* v =HEADf.get(); slice_to_heads(qkv,S,5*E,2*E, v);
        float* qd=HEADf.get(); slice_to_heads(qkv,S,5*E,3*E, qd);
        float* kd=HEADf.get(); slice_to_heads(qkv,S,5*E,4*E, kd);
        {float*t=HEADf.get(); rmsnorm(q, F32(W("s_qn")),H*S,D,t); q=t;}
        {float*t=HEADf.get(); rmsnorm(qd,F32(W("s_qn")),H*S,D,t); qd=t;}
        {float*t=HEADf.get(); rmsnorm(k, F32(W("s_kn")),H*S,D,t); k=t;}
        {float*t=HEADf.get(); rmsnorm(kd,F32(W("s_kn")),H*S,D,t); kd=t;}
        {float*t=HEADf.get(); rope(q, rc,rs,H,S,t); q=t;}
        {float*t=HEADf.get(); rope(k, rc,rs,H,S,t); k=t;}
        {float*t=HEADf.get(); rope(qd,rc,rs,H,S,t); qd=t;}
        {float*t=HEADf.get(); rope(kd,rc,rs,H,S,t); kd=t;}
        flash_i8(q,k,qd,kd,v,S,S,FL_out.data());
        float* ao=HEADf.get(); from_heads(FL_out.data(),S,ao);
        int8_t* ao_i8=QROW.get(); float* ao_s=SROWf.get(); quant_rows(ao,S,E,ao_i8,ao_s);
        gemm_i8(ao_i8,I8(W("s_out.q")),S,E,E,acc);
        deqgate(acc,ao_s,F32(W("s_out.scale")),nullptr,g_s,res,S,E,xnext);
        std::swap(x,xnext);
        // ===== cross-attn =====
        HEADf.reset();
        int8_t* hc_i8=QROW.get(); float* hc_s=SROWf.get();
        rmsmodq("rmsmodq_mod0",x,F32(W("cx_g")),nullptr,nullptr,S,E,hc_i8,hc_s);
        gemm_i8(hc_i8,I8(W("c_q.q")),S,2*E,E,acc);
        float* q2=WIDEf.get(); deq(acc,hc_s,F32(W("c_q.scale")),S,2*E,q2);
        gemm_i8(ctx_i8,I8(W("c_kv.q")),CROSS,3*E,E,acc);
        float* kv=WIDEf.get(); deq(acc,ctx_s,F32(W("c_kv.scale")),CROSS,3*E,kv);
        float* cq =HEADf.get(); slice_to_heads(q2,S,2*E,0, cq);
        float* cqd=HEADf.get(); slice_to_heads(q2,S,2*E,E, cqd);
        float* ck =HEADf.get(); slice_to_heads(kv,CROSS,3*E,0,   ck);
        float* ckd=HEADf.get(); slice_to_heads(kv,CROSS,3*E,E,   ckd);
        float* cv =HEADf.get(); slice_to_heads(kv,CROSS,3*E,2*E, cv);
        {float*t=HEADf.get(); rmsnorm(cq, F32(W("c_qn")),H*S,D,t);     cq=t;}
        {float*t=HEADf.get(); rmsnorm(cqd,F32(W("c_qn")),H*S,D,t);     cqd=t;}
        {float*t=HEADf.get(); rmsnorm(ck, F32(W("c_kn")),H*CROSS,D,t); ck=t;}
        {float*t=HEADf.get(); rmsnorm(ckd,F32(W("c_kn")),H*CROSS,D,t); ckd=t;}
        flash_i8(cq,ck,cqd,ckd,cv,S,CROSS,FL_out.data());
        float* co=HEADf.get(); from_heads(FL_out.data(),S,co);
        int8_t* co_i8=QROW.get(); float* co_s=SROWf.get(); quant_rows(co,S,E,co_i8,co_s);
        gemm_i8(co_i8,I8(W("c_out.q")),S,E,E,acc);
        deqadd(acc,co_s,F32(W("c_out.scale")),x,F32(W("local")),S,E,xnext);
        std::swap(x,xnext);
        // ===== FFN =====
        std::memcpy(RES.data(),x,(size_t)S*E*sizeof(float)); res=RES.data();
        int8_t* f_i8=QROW.get(); float* f_s=SROWf.get();
        rmsmodq("rmsmodq_mod1",x,F32(W("ff_g")),sc_f,sh_f,S,E,f_i8,f_s);
        gemm_i8(f_i8,I8(W("ff0.q")),S,2*FF,E,acc);
        int8_t* fh_i8=QROW.get(); float* fh_s=SROWf.get();
        deqglu(acc,f_s,F32(W("ff0.scale")),F32(W("ff0.bias")),S,FF,fh_i8,fh_s);
        gemm_i8(fh_i8,I8(W("ff2.q")),S,E,FF,acc);
        deqgate(acc,fh_s,F32(W("ff2.scale")),F32(W("ff2.bias")),g_f,res,S,E,xnext);
        std::swap(x,xnext);
    }
    // ===== post =====
    reset_block_pools();
    const float* xt=x+(size_t)MEM*E;                     // x[MEM:] is contiguous -> no copy
    int8_t* qx=QROW.get(); float* sx=SROWf.get(); quant_rows(xt,Mp,E,qx,sx);
    int32_t* acc=ACC.data(); gemm_i8(qx,I8("pout.q"),Mp,256,E,acc);
    float* proj=WIDEf.get(); deq(acc,sx,F32("pout.scale"),Mp,256,proj);
    std::vector<float> post((size_t)Mp*256);
    gemm_fp(proj,F32("Wpost.wt"),Mp,256,256,post.data());
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<(size_t)Mp*256;i++) post[i]+=proj[i];
    return post;
}

static double check_mad(const std::vector<float>&post,int Mp){
    float* gold=F32("out");                              // shape [1,256,Mp] row-major
    double mad=0; for(int c=0;c<256;c++)for(int mm=0;mm<Mp;mm++){
        float mine=post[(size_t)mm*256+c]; float g=gold[(size_t)c*Mp+mm]; mad=std::max(mad,(double)std::fabs(mine-g));}
    return mad;
}
// quality metrics vs the int8 golden ("out"): max_abs_diff, cosine similarity, PSNR(SNR dB).
static void check_quality(const std::vector<float>&post,int Mp,double&mad,double&cos,double&psnr){
    float* gold=F32("out");                              // [256,Mp] row-major (c-major)
    double dot=0,na=0,nb=0,se=0,sg=0; mad=0;
    for(int c=0;c<256;c++)for(int mm=0;mm<Mp;mm++){
        double mine=post[(size_t)mm*256+c], g=gold[(size_t)c*Mp+mm], e=mine-g;
        mad=std::max(mad,std::fabs(e)); dot+=mine*g; na+=mine*mine; nb+=g*g; se+=e*e; sg+=g*g;}
    cos=(na>0&&nb>0)?dot/(std::sqrt(na)*std::sqrt(nb)):0.0;
    psnr=(se>0)?10.0*std::log10(sg/se):1e9;
}

int main(int argc,char**argv){
    int threads=0, reps=0, warmup=3, L=128;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        auto nv=[&](const char*k){return a==k && i+1<argc;};
        if(nv("--threads")) threads=atoi(argv[++i]);
        else if(nv("--reps")) reps=atoi(argv[++i]);
        else if(nv("--warmup")) warmup=atoi(argv[++i]);
        else if(nv("--L")) L=atoi(argv[++i]);
        else if(nv("--gemm")){std::string g=argv[++i]; USE_ONEDNN=(g=="onednn"||g=="imm");}
        else if(nv("--core")) COREBASE=argv[++i];
        else if(nv("--flashso")){const char* pth=argv[++i];
            void* h=dlopen(pth,RTLD_NOW|RTLD_LOCAL);
            if(!h){fprintf(stderr,"dlopen flashso %s: %s\n",pth,dlerror());return 1;}
            FLASH_OVERRIDE=dlsym(h,"_flash_diff_i8_kernel");
            if(!FLASH_OVERRIDE){fprintf(stderr,"dlsym flash override failed\n");return 1;}
            printf("flash override: %s\n",pth);}
        else if(nv("--flashbm")) FLASH_BM=atoi(argv[++i]);
    }
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    if(L>0) COREBASE = std::string(sa3_home()) + "/dit_medium_cpu_amx/core_L" + std::to_string(L);

    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    #pragma omp parallel                                 // ensure every OMP thread has AMX permission
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init();
    load_core(); load_kernels();
    // Select flash kernel: BM=32 uses the built-in AOT kernel; BM in {64,128} auto-loads the
    // bit-exact wider-tile .so from so_flash/ (fewer per-tile launches). --flashso overrides.
    if(FLASH_OVERRIDE==nullptr && FLASH_BM!=32){
        std::string fp=std::string(FLASH_SO_DIR)+"/_flash_bm"+std::to_string(FLASH_BM)+".so";
        void* h=dlopen(fp.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(h) FLASH_OVERRIDE=dlsym(h,"_flash_diff_i8_kernel");
        if(!FLASH_OVERRIDE){fprintf(stderr,"WARN flash BM=%d .so unavailable (%s); falling back to built-in BM=32\n",FLASH_BM,fp.c_str()); FLASH_BM=32;}
    }
    printf("flash: BM=%d src=%s\n",FLASH_BM,FLASH_OVERRIDE?"so_flash":"builtin(cpp_kernels)");
    int S=(int)TEN.at("x_init").shp[0];                  // S from injected preamble
    int Mp=S-MEM;
    alloc_scratch(S);
    printf("gemm=%s threads=%d L=%d S=%d Mp=%d dnnl_isa=%s\n",
           USE_ONEDNN?"onednn":"triton",threads,L,S,Mp,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));

    if(reps<=0){                                         // correctness-only
        auto post=run_forward(S); double mad,cos,psnr; check_quality(post,Mp,mad,cos,psnr);
        bool ok=(cos>=0.9999);
        printf("[check] gemm=%s max_abs_diff=%.6e  cos=%.7f  psnr=%.1fdB  bit_exact=%s  quality_ok(cos>=0.9999)=%s\n",
               USE_ONEDNN?"onednn":"triton",mad,cos,psnr,mad==0.0?"YES":"no",ok?"YES":"NO");
        return ok?0:2;
    }
    // timing mode
    std::vector<double> ms; double mad0=-1;
    for(int r=0;r<warmup+reps;r++){
        auto t0=std::chrono::steady_clock::now();
        auto post=run_forward(S);
        auto t1=std::chrono::steady_clock::now();
        double m=std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0;
        if(r==0) mad0=check_mad(post,Mp);
        if(r>=warmup) ms.push_back(m);
    }
    std::sort(ms.begin(),ms.end());
    double med=ms[ms.size()/2], mn=ms.front();
    printf("[time] gemm=%s threads=%d L=%d  per-forward median=%.1f ms  min=%.1f ms  (reps=%d warmup=%d)  bitexact_mad=%.3e\n",
           USE_ONEDNN?"onednn":"triton",threads,L,med,mn,reps,warmup,mad0);
    return 0;
}
