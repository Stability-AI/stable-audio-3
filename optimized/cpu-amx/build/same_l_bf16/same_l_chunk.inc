// same_l_chunk.inc — chunked / chunk-parallel decode (the PRIMARY SAME-L decode path).
// Included at the end of same_l_cpu_amx.cpp (shares all statics).
//
// SAME-L uses a banded SWA mask (window +-17 internal tokens) => the receptive field over 12
// blocks is a few latent tokens. Tiling [1,256,T] into C-latent interiors with `overlap` latent
// tokens of context each side and concatenating the interior audio reproduces the whole decode
// to high PSNR. Overlap floor is 8 (campaign: ovl 6->64 dB, 8->82, plateau). C=64 is the ship size.
// (Unlike the TFLite dense-[S,S]-mask decoder this dodges no quadratic — the C++ band attention is
//  already linear — but chunking bounds working-set/RoPE and enables chunk-parallelism.)

struct ChunkL{int a,b,lo,hi;};
static std::vector<ChunkL> chunk_plan_l(int T,int C,int overlap){
    std::vector<ChunkL> plan; int a=0;
    while(a<T){
        int b=std::min(a+C,T);
        int lo=std::max(a-overlap,0), hi=std::min(b+overlap,T);
        plan.push_back({a,b,lo,hi}); a=b;
    }
    return plan;
}

extern "C" {
// Chunked decode. C = interior chunk width (latent tokens); overlap ctx each side (use 8).
// parallel=0: chunks run sequentially, each using all threads (cache-blocking).
// parallel=1: independent chunks run in parallel (1 thread each), across cores.
// out_patches [1,512,16T] caller-allocated.
void samel_forward_chunked(const float* latent,int T,int C,int overlap,int parallel,float* out_patches){
    if(C<=0) C=64; if(overlap<0) overlap=8;
    std::vector<ChunkL> plan=chunk_plan_l(T,C,overlap);
    int nch=(int)plan.size(); int L=SIN*T;
    if(!parallel){
        static std::vector<float> sublat, subpat;
        for(auto&ck:plan){
            int w=ck.hi-ck.lo;
            sublat.assign((size_t)LAT*w,0);
            for(int c=0;c<LAT;c++)for(int t=ck.lo;t<ck.hi;t++) sublat[(size_t)c*w+(t-ck.lo)]=latent[(size_t)c*T+t];
            subpat.assign((size_t)OUTCH*SIN*w,0);
            decode(0,true,sublat.data(),w,subpat.data());
            int swid=SIN*w, off=(ck.a-ck.lo)*SIN, ow=(ck.b-ck.a)*SIN, ocol=ck.a*SIN;
            for(int ch=0;ch<OUTCH;ch++)
                std::memcpy(out_patches+(size_t)ch*L+ocol, subpat.data()+(size_t)ch*swid+off, ow*sizeof(float));
        }
        return;
    }
    int nw=(int)AR.size()-1; if(nw<1) nw=1;
    #pragma omp parallel for schedule(dynamic) num_threads(nw)
    for(int i=0;i<nch;i++){
        int slot=omp_get_thread_num()+1;
        ChunkL ck=plan[i]; int w=ck.hi-ck.lo;
        std::vector<float> sublat((size_t)LAT*w), subpat((size_t)OUTCH*SIN*w);
        for(int c=0;c<LAT;c++)for(int t=ck.lo;t<ck.hi;t++) sublat[(size_t)c*w+(t-ck.lo)]=latent[(size_t)c*T+t];
        decode(slot,false,sublat.data(),w,subpat.data());
        int swid=SIN*w, off=(ck.a-ck.lo)*SIN, ow=(ck.b-ck.a)*SIN, ocol=ck.a*SIN;
        for(int ch=0;ch<OUTCH;ch++)
            std::memcpy(out_patches+(size_t)ch*L+ocol, subpat.data()+(size_t)ch*swid+off, ow*sizeof(float));
    }
}
} // extern "C"
