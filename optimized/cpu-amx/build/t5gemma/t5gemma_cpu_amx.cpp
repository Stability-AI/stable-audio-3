// t5gemma_cpu_amx.cpp — torch-free C++ AMX-BF16 T5Gemma ENCODER as a callable .so.
//
// google/t5gemma-b-b-ul2 encoder half (Gemma2-style): 12 layers, dim=768, 12 heads,
// head_dim=64, GeGLU(2048), RMSNorm(1+w) sandwich, RoPE theta=10000 half-half,
// attn logit softcap=50, embed x sqrt(768). Seq FIXED at 256 (pad token id 0).
// STANDARD softmax attention (not differential/chunked) — simpler than the decoders.
//
// Mirrors the proven SAME-L / SAME-S engines: oneDNN AMX-BF16 GEMMs for every linear
// (q,k,v,o,gate,up,down + the embed gather table are bf16). RMSNorm, RoPE, softcap-tanh,
// softmax, GeLU and residuals stay fp32 (AVX-512 intrinsics for exp/tanh). The attention
// QK^T and P@V are ALSO bf16-AMX batched matmuls (per-head) — T5Gemma uses STANDARD softmax
// attention (not SAME-L's cancellation-fragile differential attn), so the softcap+softmax
// fp32 island absorbs the bf16 rounding: 12-token gate prompts stay 62-67 dB / cos>=0.9997.
// (Scores + softmax are computed in fp32 between the two bf16 matmuls.)
//
//   t5g_init(weights_base, threads)                        -> mmap bf16 weights + AMX/omp/oneDNN
//   t5g_forward(ids[256] i32, mask[256] i32, out[256*768]) -> last_hidden_state fp32
//
// build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I$ONEINC \
//            t5gemma_cpu_amx.cpp -o t5gemma_cpu_amx.so $ONELIB/libdnnl.a -ldl -lpthread -lm
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
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

// ── architecture constants (t5gemma-b-b-ul2 encoder) ──
static const int S=256, DIM=768, H=12, HD=64, HALF=32, NB=12, FF=2048;
static const float SOFTCAP=50.0f, SCALE=0.125f, EPS=1e-6f;   // SCALE=64**-0.5
static float EMBED_SCALE=0.0f;                                // sqrt(768), set in init

// ── optional phase profiler (T5G_PROF=1) ──
static double PROF[8]={0};
static const char* PROFLBL[8]={"gemm","attn","norm","rope","glu","cast","misc",""};
static bool PROF_ON=false;
static inline double wt(){return omp_get_wtime();}
#define TB(id) do{ if(PROF_ON){double _n=wt(); PROF[id]+=_n-_pt; _pt=_n;} }while(0)

typedef uint16_t bf16;
static inline bf16 f2b(float f){ // round-to-nearest-even f32->bf16
    uint32_t x; std::memcpy(&x,&f,4);
    uint32_t r=x+0x7fff+((x>>16)&1); return (bf16)(r>>16);
}
static inline float b2f(bf16 h){ uint32_t x=(uint32_t)h<<16; float f; std::memcpy(&f,&x,4); return f; }

// ── fast vectorizable transcendentals (pure float -> AVX-512 auto-vec; err ~2e-6 << bf16) ──
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
// gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))
static inline float vgelu(float x){
    float x3=x*x*x;
    return 0.5f*x*(1.0f+vtanh(0.7978845608028654f*(x+0.044715f*x3)));
}

#if defined(__AVX512F__)
// AVX-512 exp/tanh — SAME polynomial as scalar vexp() (numerically identical), but the
// 16-lane transcendentals actually vectorize (the scalar vexp/vtanh do NOT auto-vec inside
// omp-simd because of floor/bitcast). This is the attention hot-path lever.
static inline __m512 exp512(__m512 x){
    x=_mm512_max_ps(x,_mm512_set1_ps(-87.0f)); x=_mm512_min_ps(x,_mm512_set1_ps(88.0f));
    __m512 z=_mm512_mul_ps(x,_mm512_set1_ps(1.442695041f));
    __m512 n=_mm512_roundscale_ps(z,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512 f=_mm512_sub_ps(z,n);
    __m512 p=_mm512_set1_ps(0.0013333f);
    p=_mm512_fmadd_ps(p,f,_mm512_set1_ps(0.0096181f));
    p=_mm512_fmadd_ps(p,f,_mm512_set1_ps(0.0555041f));
    p=_mm512_fmadd_ps(p,f,_mm512_set1_ps(0.2402265f));
    p=_mm512_fmadd_ps(p,f,_mm512_set1_ps(0.6931472f));
    p=_mm512_fmadd_ps(p,f,_mm512_set1_ps(1.0f));
    __m512i ni=_mm512_add_epi32(_mm512_cvtps_epi32(n),_mm512_set1_epi32(127));
    __m512 s=_mm512_castsi512_ps(_mm512_slli_epi32(ni,23));
    return _mm512_mul_ps(p,s);
}
static inline __m512 tanh512(__m512 x){
    __m512 e=exp512(_mm512_mul_ps(x,_mm512_set1_ps(2.0f)));
    return _mm512_sub_ps(_mm512_set1_ps(1.0f),
                         _mm512_div_ps(_mm512_set1_ps(2.0f),_mm512_add_ps(e,_mm512_set1_ps(1.0f))));
}
#endif

// ------------------------- mmap weights.bin + manifest (verbatim from SAME-L) -------------------------
struct Ten{void* p; std::string dt; long n; std::vector<long> shp;};
static std::map<std::string,Ten> TEN;
static char* BASE=nullptr;

// Engine paths resolve from $SA3_CPUAMX_HOME (same base the Python side uses), so nothing
// absolute is baked into the binary. Falls back to the current directory.
static const char* sa3_home() {
    const char* v = getenv("SA3_CPUAMX_HOME");
    return (v && *v) ? v : ".";
}
static std::string WBASE = std::string(sa3_home()) + "/t5gemma_cpu_amx/weights";
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
    printf("[t5g] weights mmap'd: %ld arrays (%s)\n",(long)TEN.size(),bin.c_str());
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

// ------------------------- RoPE table (half-half, positions 0..S-1, 32 freqs) -------------------------
static std::vector<float> RCOS, RSIN;   // [S*HALF]
static void rope_build(){
    const float* inv=F32("rope_inv");   // (32,) fp32 from npz
    RCOS.resize((size_t)S*HALF); RSIN.resize((size_t)S*HALF);
    for(int p=0;p<S;p++) for(int i=0;i<HALF;i++){
        double ang=(double)p*(double)inv[i];
        RCOS[(size_t)p*HALF+i]=(float)std::cos(ang);
        RSIN[(size_t)p*HALF+i]=(float)std::sin(ang);
    }
}

// ------------------------- reusable arena (S fixed at 256) -------------------------
struct Arena{
    std::vector<float> x,h,q,k,v,ao,gate,up,scores,outh;   // f32 activations
    std::vector<bf16>  srcb, qg,kgT,vg,pb;       // bf16 staging (GEMM src + packed attn operands)
    void init(){
        x.assign((size_t)S*DIM,0);  h.assign((size_t)S*DIM,0);
        q.assign((size_t)S*DIM,0);  k.assign((size_t)S*DIM,0);  v.assign((size_t)S*DIM,0);
        ao.assign((size_t)S*DIM,0); gate.assign((size_t)S*FF,0); up.assign((size_t)S*FF,0);
        scores.assign((size_t)H*S*S,0); outh.assign((size_t)H*S*HD,0);
        srcb.assign((size_t)S*FF,0);
        qg.assign((size_t)H*S*HD,0); kgT.assign((size_t)H*HD*S,0);
        vg.assign((size_t)H*S*HD,0); pb.assign((size_t)H*S*S,0);
    }
};
static Arena A;

// ------------------------- fp32 elementwise kernels -------------------------
// Gemma RMSNorm: n = x/sqrt(mean(x^2)+eps); out = n*(1+w). Accumulate in double (matches numpy ref).
static void rmsnorm(const float* x,float* o,const float* w,int M){
    #pragma omp parallel for schedule(static)
    for(int m=0;m<M;m++){
        const float* xr=x+(size_t)m*DIM; float* orr=o+(size_t)m*DIM;
        double ss=0; for(int j=0;j<DIM;j++) ss+=(double)xr[j]*xr[j];
        double inv=1.0/std::sqrt(ss/DIM+(double)EPS);
        for(int j=0;j<DIM;j++) orr[j]=(float)((double)xr[j]*inv*(1.0+(double)w[j]));
    }
}
static void resadd(float* x,const float* y,int M){
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<(size_t)M*DIM;i++) x[i]+=y[i];
}
static void to_bf16(const float* x,bf16* o,size_t n){
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;i++) o[i]=f2b(x[i]);
}
// RoPE in place on a [M,DIM] tensor (H heads x HD), position p=m. Pairs (i, i+HALF).
static void rope(float* r,int M){
    #pragma omp parallel for schedule(static)
    for(int m=0;m<M;m++){
        const float* cs=&RCOS[(size_t)m*HALF]; const float* sn=&RSIN[(size_t)m*HALF];
        float* row=r+(size_t)m*DIM;
        for(int h=0;h<H;h++){
            float* hd=row+h*HD;
            for(int i=0;i<HALF;i++){
                float a=hd[i], b=hd[i+HALF], c=cs[i], s=sn[i];
                hd[i]=a*c-b*s; hd[i+HALF]=b*c+a*s;
            }
        }
    }
}

// ------------------------- standard softmax attention (fp32) ---------
// out[i] = softmax_j( SOFTCAP*tanh((q_i . k_j)*SCALE / SOFTCAP) + addm[j] ) @ v.  full seq.
// QK^T and P@V are batched f32 matmuls via oneDNN (batch=H heads), using TRANSPOSED-STRIDE
// memory views straight over the [S,DIM] q/k/v tensors (each head = a [S,HD] sub-block, no
// gather/copy). softcap-tanh + softmax run in between on scores[H,S,S] with 16-lane intrinsics.
static const float SC_ARG=SCALE/SOFTCAP;
static void attention(const float* q,const float* k,const float* v,const float* addm,float* ao){
    using dt=dnnl::memory::data_type; using md=dnnl::memory::desc;
    float* scores=A.scores.data(); float* outh=A.outh.data();
    bf16 *qg=A.qg.data(), *kgT=A.kgT.data(), *vg=A.vg.data(), *pb=A.pb.data();
    dnnl::stream& strm=*MMC[0].strm;

    // ---- gather+cast per-head PACKED bf16 operands (AMX-friendly, standard layouts) ----
    // qg[h,i,:]=q[i,h*HD:], vg[h,i,:]=v[i,h*HD:]  ([H,S,HD]);  kgT[h,:,j]=k[j,h*HD:] ([H,HD,S])
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++) for(int i=0;i<S;i++){
        const float* qr=q+(size_t)i*DIM+h*HD; const float* vr=v+(size_t)i*DIM+h*HD;
        bf16* dq=qg+((size_t)h*S+i)*HD; bf16* dv=vg+((size_t)h*S+i)*HD;
        for(int t=0;t<HD;t++){ dq[t]=f2b(qr[t]); dv[t]=f2b(vr[t]); }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++) for(int j=0;j<S;j++){
        const float* kr=k+(size_t)j*DIM+h*HD; bf16* base=kgT+(size_t)h*HD*S+j;
        for(int d=0;d<HD;d++) base[(size_t)d*S]=f2b(kr[d]);
    }

    // ---- QK^T (bf16 AMX): scores[h,i,j] = sum_d qg[h,i,d]*kgT[h,d,j] ----
    static dnnl::matmul qk_p; static dnnl::memory qk_a,qk_b,qk_c; static bool qk_i=false;
    if(!qk_i){
        md a_md({H,S,HD},dt::bf16,{(long)S*HD,HD,1});   // qg  [H,S,HD] packed
        md b_md({H,HD,S},dt::bf16,{(long)HD*S,S,1});    // kgT [H,HD,S] packed
        md c_md({H,S,S}, dt::f32, {(long)S*S,S,1});
        dnnl::matmul::primitive_desc pd(*ENG,a_md,b_md,c_md);
        qk_p=dnnl::matmul(pd);
        qk_a=dnnl::memory(pd.src_desc(),*ENG,(void*)qg);
        qk_b=dnnl::memory(pd.weights_desc(),*ENG,(void*)kgT);
        qk_c=dnnl::memory(pd.dst_desc(),*ENG,(void*)scores);
        qk_i=true;
    }
    qk_a.set_data_handle((void*)qg); qk_b.set_data_handle((void*)kgT); qk_c.set_data_handle((void*)scores);
    qk_p.execute(strm,{{DNNL_ARG_SRC,qk_a},{DNNL_ARG_WEIGHTS,qk_b},{DNNL_ARG_DST,qk_c}}); strm.wait();

    // ---- softcap + additive mask + softmax on each of the H*S score rows; write pb=bf16(P) ----
    #pragma omp parallel for schedule(static)
    for(int r=0;r<H*S;r++){
        float* sr=scores+(size_t)r*S; bf16* pr=pb+(size_t)r*S;
        float mx,z;
#if defined(__AVX512F__)
        __m512 vmax=_mm512_set1_ps(-1e30f);
        const __m512 vsc=_mm512_set1_ps(SC_ARG), vcap=_mm512_set1_ps(SOFTCAP);
        for(int j=0;j<S;j+=16){
            __m512 s=tanh512(_mm512_mul_ps(_mm512_loadu_ps(sr+j),vsc));
            s=_mm512_fmadd_ps(s,vcap,_mm512_loadu_ps(addm+j));
            _mm512_storeu_ps(sr+j,s); vmax=_mm512_max_ps(vmax,s);
        }
        mx=_mm512_reduce_max_ps(vmax);
        __m512 vz=_mm512_setzero_ps(); const __m512 vmx=_mm512_set1_ps(mx);
        for(int j=0;j<S;j+=16){
            __m512 e=exp512(_mm512_sub_ps(_mm512_loadu_ps(sr+j),vmx));
            _mm512_storeu_ps(sr+j,e); vz=_mm512_add_ps(vz,e);
        }
        z=_mm512_reduce_add_ps(vz);
        const __m512 viz=_mm512_set1_ps(1.0f/z);
        for(int j=0;j<S;j+=16){
            __m512 p=_mm512_mul_ps(_mm512_loadu_ps(sr+j),viz);   // normalize
            // f32->bf16 round-to-nearest-even, 16-lane -> pb
            __m512i bits=_mm512_castps_si512(p);
            __m512i lsb=_mm512_and_si512(_mm512_srli_epi32(bits,16),_mm512_set1_epi32(1));
            bits=_mm512_add_epi32(_mm512_add_epi32(bits,_mm512_set1_epi32(0x7fff)),lsb);
            _mm256_storeu_si256((__m256i*)(pr+j),_mm512_cvtepi32_epi16(_mm512_srli_epi32(bits,16)));
        }
#else
        for(int j=0;j<S;j++) sr[j]=SOFTCAP*vtanh(sr[j]*SC_ARG)+addm[j];
        mx=-1e30f; for(int j=0;j<S;j++) mx=sr[j]>mx?sr[j]:mx;
        z=0; for(int j=0;j<S;j++){ float e=vexp(sr[j]-mx); sr[j]=e; z+=e; }
        float iz=1.0f/z; for(int j=0;j<S;j++) pr[j]=f2b(sr[j]*iz);
#endif
    }

    // ---- P@V (bf16 AMX): outh[h,i,d] = sum_j pb[h,i,j]*vg[h,j,d] ----
    static dnnl::matmul pv_p; static dnnl::memory pv_a,pv_b,pv_c; static bool pv_i=false;
    if(!pv_i){
        md a_md({H,S,S}, dt::bf16,{(long)S*S,S,1});    // pb [H,S,S] packed
        md b_md({H,S,HD},dt::bf16,{(long)S*HD,HD,1});  // vg [H,S,HD] packed
        md c_md({H,S,HD},dt::f32, {(long)S*HD,HD,1});  // outh [H,S,HD] packed
        dnnl::matmul::primitive_desc pd(*ENG,a_md,b_md,c_md);
        pv_p=dnnl::matmul(pd);
        pv_a=dnnl::memory(pd.src_desc(),*ENG,(void*)pb);
        pv_b=dnnl::memory(pd.weights_desc(),*ENG,(void*)vg);
        pv_c=dnnl::memory(pd.dst_desc(),*ENG,(void*)outh);
        pv_i=true;
    }
    pv_a.set_data_handle((void*)pb); pv_b.set_data_handle((void*)vg); pv_c.set_data_handle((void*)outh);
    pv_p.execute(strm,{{DNNL_ARG_SRC,pv_a},{DNNL_ARG_WEIGHTS,pv_b},{DNNL_ARG_DST,pv_c}}); strm.wait();

    // ---- scatter outh[H,S,HD] -> ao[S, h*HD] ----
    #pragma omp parallel for collapse(2) schedule(static)
    for(int h=0;h<H;h++) for(int i=0;i<S;i++)
        std::memcpy(ao+(size_t)i*DIM+h*HD, outh+((size_t)h*S+i)*HD, HD*4);
}

// ------------------------- one encoder layer -------------------------
static void run_layer(int l,const float* addm){
    auto W=[&](const char* n){ return "L"+std::to_string(l)+"."+std::string(n); };
    float* x=A.x.data(); bf16* srcb=A.srcb.data();
    double _pt=wt();
    // ---- attention sub-block (sandwich norm) ----
    rmsnorm(x, A.h.data(), F32(W("pre_a")), S); TB(2);          // h = pre_self_attn_norm(x)
    to_bf16(A.h.data(), srcb, (size_t)S*DIM); TB(5);
    gemm_bf16(0, srcb, BF(W("q")), S, DIM, DIM, A.q.data());
    gemm_bf16(0, srcb, BF(W("k")), S, DIM, DIM, A.k.data());
    gemm_bf16(0, srcb, BF(W("v")), S, DIM, DIM, A.v.data()); TB(0);
    rope(A.q.data(), S); rope(A.k.data(), S); TB(3);
    attention(A.q.data(), A.k.data(), A.v.data(), addm, A.ao.data()); TB(1);   // ao = attn(...)
    to_bf16(A.ao.data(), srcb, (size_t)S*DIM); TB(5);
    gemm_bf16(0, srcb, BF(W("o")), S, DIM, DIM, A.q.data()); TB(0);   // q(tmp) = o_proj(ao)
    rmsnorm(A.q.data(), A.h.data(), F32(W("post_a")), S); TB(2);      // h = post_self_attn_norm(tmp)
    resadd(x, A.h.data(), S); TB(6);                                  // x += h
    // ---- FFN sub-block (GeGLU, sandwich norm) ----
    rmsnorm(x, A.h.data(), F32(W("pre_f")), S); TB(2);               // h = pre_ffn_norm(x)
    to_bf16(A.h.data(), srcb, (size_t)S*DIM); TB(5);
    gemm_bf16(0, srcb, BF(W("gate")), S, FF, DIM, A.gate.data());
    gemm_bf16(0, srcb, BF(W("up")),   S, FF, DIM, A.up.data()); TB(0);
    // fuse GeGLU: glu = gelu(gate)*up -> bf16 for down GEMM
    #pragma omp parallel for schedule(static)
    for(int m=0;m<S;m++){
        const float* g=A.gate.data()+(size_t)m*FF; const float* u=A.up.data()+(size_t)m*FF;
        bf16* r=srcb+(size_t)m*FF;
        for(int j=0;j<FF;j++) r[j]=f2b(vgelu(g[j])*u[j]);
    } TB(4);
    gemm_bf16(0, srcb, BF(W("down")), S, DIM, FF, A.q.data()); TB(0); // q(tmp) = down_proj(glu)
    rmsnorm(A.q.data(), A.h.data(), F32(W("post_f")), S); TB(2);      // h = post_ffn_norm(tmp)
    resadd(x, A.h.data(), S); TB(6);                                  // x += h
}

// ============================== C ABI ==============================
static int NTHREADS=16;
extern "C" {

int t5g_init(const char* weights_base,int threads){
    if(threads<=0){const char* e=getenv("OMP_NUM_THREADS"); threads=e?atoi(e):16;}
    NTHREADS=threads;
    if(weights_base && weights_base[0]) WBASE=weights_base;
    if(syscall(SYS_arch_prctl,0x1023,18)!=0){fprintf(stderr,"[t5g] arch_prctl AMX FAILED\n");return 1;}
    omp_set_dynamic(0); omp_set_num_threads(threads);
    omp_set_max_active_levels(1);
    #pragma omp parallel num_threads(threads)
    { syscall(SYS_arch_prctl,0x1023,18); }
    onednn_init(); mmcache_init(threads);
    load_weights(); rope_build();
    EMBED_SCALE=(float)std::sqrt((double)DIM);
    A.init();
    PROF_ON = getenv("T5G_PROF")!=nullptr;
    printf("[t5g] init ok: threads=%d isa=%s embed_scale=%.6f\n",
           threads,dnnl_cpu_isa2str(dnnl_get_effective_cpu_isa()),EMBED_SCALE);
    fflush(stdout);
    return 0;
}

// forward: input_ids[256] int32, attention_mask[256] int32 (1=real,0=pad)
//          -> out last_hidden_state[256*768] f32 (caller-allocated)
void t5g_forward(const int32_t* ids,const int32_t* mask,float* out){
    // embedding gather (bf16 table) + x sqrt(768)
    bf16* emb=BF("embed");
    #pragma omp parallel for schedule(static)
    for(int s=0;s<S;s++){
        const bf16* row=emb+(size_t)ids[s]*DIM;
        float* xr=A.x.data()+(size_t)s*DIM;
        for(int c=0;c<DIM;c++) xr[c]=b2f(row[c])*EMBED_SCALE;
    }
    // additive mask on key axis: (1-keep)*-1e9
    static float addm[S];
    for(int j=0;j<S;j++) addm[j]=(1.0f-(float)mask[j])*-1e9f;

    for(int l=0;l<NB;l++) run_layer(l,addm);

    rmsnorm(A.x.data(), out, F32("norm"), S);   // final norm -> caller buffer
}

int t5g_DIM(){return DIM;} int t5g_S(){return S;}

void t5g_prof_dump(){
    double tot=0; for(int i=0;i<7;i++) tot+=PROF[i];
    printf("[prof] ");
    for(int i=0;i<7;i++) printf("%s=%.1fms(%.0f%%) ",PROFLBL[i],PROF[i]*1e3,tot>0?100*PROF[i]/tot:0);
    printf(" total=%.1fms\n",tot*1e3);
    for(int i=0;i<8;i++) PROF[i]=0; fflush(stdout);
}

} // extern "C"
