// dit_cpu_amx.cpp — torch-free C++ AMX int8 SA3-medium DiT as a PER-STEP CALLABLE .so.
// Refactor of dit_amx_forward3.cpp (the proven, bit-exact standalone driver): the 24-block
// int8/AMX forward is kept byte-identical, but the CLI/timing main() is replaced by a small
// C ABI so the core is callable per sampling step:
//     dit_init(core_bin_base, threads)  -> load int8 block weights ONCE + AMX/omp/oneDNN/kernels
//     dit_forward(x_init, ctx_i8, ctx_s, gc, rope_cos, rope_sin, S, out_post)  -> denoised core
// The block/pout/Wpost weights are sequence-length-INDEPENDENT, so ANY core_L*.bin serves as the
// weight source; the per-step preamble tensors (x_init/ctx/gc) and the length-dependent rope are
// supplied by the caller (numpy preamble). Kernels are shape-general (proven L128 .so runs any L).
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            dit_cpu_amx.cpp -o dit_cpu_amx.so $ONELIB/libdnnl.a -ldl -lpthread -lm
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
static const char* AOT="/weka2/cj/clod/tritoncpu_sa3/aot_stage2";        // so/ + cpp_kernels.txt (L-independent)
static std::string COREBASE="/weka2/cj/clod/tritoncpu_sa3/aot_speedprove/core_L320";  // .bin + _manifest.txt (weights)

static bool USE_ONEDNN=true;                     // int8 linear GEMM backend
static void* FLASH_OVERRIDE=nullptr;             // optional alternate flash kernel .so entry
static int   FLASH_BM=128;                       // flash query-block size (BM); constexpr in the .so.
static const char* FLASH_SO_DIR="/weka2/cj/clod/tritoncpu_sa3/aot_speedprove/so_flash";

static inline int cdiv(int a,int b){return (a+b-1)/b;}
static int npow2(int n){int p=1;while(p<n)p<<=1;return p;}
typedef unsigned u32;

// ------------------------- mmap core.bin + manifest (int8 block weights) -------------------------
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
    printf("[dit_cpu_amx] core weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
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
    printf("[dit_cpu_amx] dlopen'd %ld AOT kernels\n",(long)FN.size());
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

// ------------------------- reusable scratch pools (allocated once per S, no zero-init) -------------------------
template<class T> struct Pool{
    std::vector<std::vector<T>> slab; size_t elems=0; int cur=0;
    void init(int n,size_t e){ elems=e; slab.assign(n,std::vector<T>(e)); cur=0; }
    T* get(){ if(cur>=(int)slab.size()){fprintf(stderr,"POOL OVERFLOW (%d slabs)\n",(int)slab.size());exit(3);} return slab[cur++].data(); }
    void reset(){ cur=0; }
};
static Pool<float>   HEADf;
static Pool<float>   WIDEf;
static Pool<int8_t>  QROW;
static Pool<float>   SROWf;
static std::vector<int32_t> ACC;
static std::vector<float>   XA,XB,RES,SSG;
static std::vector<int8_t>  FL_qmi,FL_kmi,FL_qdi,FL_kdi,FL_viq;
static std::vector<float>   FL_out,FL_qsm,FL_ksm,FL_qsd,FL_ksd,FL_vsc;

static void alloc_scratch(int S){
    int Sm=std::max(S,CROSS); size_t se=(size_t)S*E, sme=(size_t)Sm*E, hsm=(size_t)H*Sm;
    HEADf.init(16, sme);
    WIDEf.init(4,  (size_t)S*5*E);
    QROW .init(8,  (size_t)S*FF);
    SROWf.init(8,  (size_t)Sm);
    ACC.assign((size_t)S*2*FF,0);
    XA.assign(se,0); XB.assign(se,0); RES.assign(se,0); SSG.assign((size_t)6*E,0);
    FL_qmi.assign(sme,0);FL_kmi.assign(sme,0);FL_qdi.assign(sme,0);FL_kdi.assign(sme,0);FL_viq.assign(sme,0);
    FL_out.assign(sme,0);FL_qsm.assign(hsm,0);FL_ksm.assign(hsm,0);FL_qsd.assign(hsm,0);FL_ksd.assign(hsm,0);FL_vsc.assign(hsm,0);
    printf("[dit_cpu_amx] scratch alloc S=%d Sm=%d (HEADf %.1fMB, ACC %.1fMB)\n",
           S,Sm,sme*4/1e6,(double)S*2*FF*4/1e6);
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
static void gemm_i8_onednn(const int8_t*a,const int8_t*b,int M,int N,int K,int32_t* dst){
    using dt=dnnl::memory::data_type;
    MMKey key{M,N,K}; auto it=MM.find(key);
    if(it==MM.end()){
        dnnl::memory::desc a_md({M,K},dt::s8, {K,1});
        dnnl::memory::desc b_md({K,N},dt::s8, {N,1});
        dnnl::memory::desc c_md({M,N},dt::s32,{N,1});
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
    std::memset(dst,0,(size_t)M*N*sizeof(int32_t)); auto fn=(t_gemm_i8)FN.at("gemm_i8");
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
static void slice_to_heads(const float*src,int S,int total,int coloff,float* o){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++)for(int h=0;h<H;h++)
        std::memcpy(o+((size_t)h*S+s)*D, src+(size_t)s*total+coloff+h*D, D*sizeof(float));
}
static void from_heads(const float*src,int S,float* o){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++)for(int s=0;s<S;s++)
        std::memcpy(o+(size_t)s*E+h*D, src+((size_t)h*S+s)*D, D*sizeof(float));
}

// ------------------------- one whole forward (24-block int8/AMX core + int8 proj + fp32 post) ------
// Caller supplies the fp32 preamble (x_init/ctx/gc) and the length-dependent rope. Writes
// out_post [Mp,256] row-major (Mp=S-MEM). Block/pout/Wpost weights come from the mmap'd core.
static void run_forward(int S,const float* x_init_p,const int8_t* ctx_i8_p,const float* ctx_s_p,
                        const float* gc_p,const float* rc,const float* rs,float* out_post){
    int Mp=S-MEM;
    float* x=XA.data(); float* xnext=XB.data();
    std::memcpy(x,x_init_p,(size_t)S*E*sizeof(float));
    const int8_t* ctx_i8=ctx_i8_p; const float* ctx_s=ctx_s_p; const float* gc=gc_p;
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
    const float* xt=x+(size_t)MEM*E;
    int8_t* qx=QROW.get(); float* sx=SROWf.get(); quant_rows(xt,Mp,E,qx,sx);
    int32_t* acc=ACC.data(); gemm_i8(qx,I8("pout.q"),Mp,256,E,acc);
    float* proj=WIDEf.get(); deq(acc,sx,F32("pout.scale"),Mp,256,proj);
    gemm_fp(proj,F32("Wpost.wt"),Mp,256,256,out_post);
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<(size_t)Mp*256;i++) out_post[i]+=proj[i];
}

// ============================== C ABI ==============================
static int CUR_S=-1;
extern "C" {

// Load int8 block weights ONCE from core_base (".bin"+"_manifest.txt"), enable AMX on every OMP
// thread, init oneDNN + dlopen the AOT kernels + the BM=128 flash kernel. threads<=0 -> env/16.
int dit_init(const char* core_base, int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    if(core_base && core_base[0]) COREBASE=core_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[dit_cpu_amx] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    #pragma omp parallel
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init();
    load_core(); load_kernels();
    if(FLASH_OVERRIDE==nullptr && FLASH_BM!=32){
        std::string fp=std::string(FLASH_SO_DIR)+"/_flash_bm"+std::to_string(FLASH_BM)+".so";
        void* h=dlopen(fp.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(h) FLASH_OVERRIDE=dlsym(h,"_flash_diff_i8_kernel");
        if(!FLASH_OVERRIDE){fprintf(stderr,"[dit_cpu_amx] WARN flash BM=%d .so unavailable; using built-in BM=32\n",FLASH_BM);FLASH_BM=32;}
    }
    printf("[dit_cpu_amx] init ok: threads=%d gemm=%s flash_bm=%d isa=%s\n",
           threads,USE_ONEDNN?"onednn":"triton",FLASH_BM,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()));
    fflush(stdout);
    return 0;
}

// One 24-block forward. Pointers are caller-owned (void* for ctypes friendliness):
//   x_init  fp32 [S*E]        ctx_i8 int8 [CROSS*E]   ctx_s fp32 [CROSS]   gc fp32 [6*E]
//   rope_cos/rope_sin fp32 [S*RD]   out_post fp32 [(S-MEM)*256] (row-major, caller-allocated)
void dit_forward(void* x_init,void* ctx_i8,void* ctx_s,void* gc,
                 void* rope_cos,void* rope_sin,int S,void* out_post){
    if(S!=CUR_S){ alloc_scratch(S); CUR_S=S; }
    run_forward(S,(const float*)x_init,(const int8_t*)ctx_i8,(const float*)ctx_s,
                (const float*)gc,(const float*)rope_cos,(const float*)rope_sin,(float*)out_post);
}

// Expose constants so the python side can sanity-check its preamble shapes.
int dit_E(){return E;} int dit_MEM(){return MEM;} int dit_CROSS(){return CROSS;} int dit_RD(){return RD;}

} // extern "C"
