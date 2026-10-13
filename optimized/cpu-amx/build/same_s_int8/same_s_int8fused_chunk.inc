// same_s_int8fused_chunk.inc — cache-blocked / chunk-parallel decode (FUSED int8 engine).
// Included at the end of same_s_int8fused_cpu_amx.cpp (shares all statics). Position-invariant
// tiling (receptive field = 2 latent tokens); identical to the naive int8 engine's chunk .inc
// except it calls the FUSED decode(). Quality-neutral vs whole (per-chunk int8 requant only).

struct Chunk{int a,b,lo,hi;};
static std::vector<Chunk> chunk_plan(int T,int C,int overlap){
    std::vector<Chunk> plan; int a=0;
    while(a<T){
        int b=std::min(a+C,T);
        int lo=std::max(a-overlap,0), hi=std::min(b+overlap,T);
        if((hi-lo)%2!=0){ if(hi<T) hi++; else lo--; }   // keep decoded width even (PAD_MODULO=2)
        plan.push_back({a,b,lo,hi}); a=b;
    }
    return plan;
}

extern "C" {
void sames_forward_chunked(const float* latent,int T,int C,int overlap,int parallel,float* out_patches){
    std::vector<Chunk> plan=chunk_plan(T,C,overlap);
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
        Chunk ck=plan[i]; int w=ck.hi-ck.lo;
        std::vector<float> sublat((size_t)LAT*w), subpat((size_t)OUTCH*SIN*w);
        for(int c=0;c<LAT;c++)for(int t=ck.lo;t<ck.hi;t++) sublat[(size_t)c*w+(t-ck.lo)]=latent[(size_t)c*T+t];
        decode(slot,false,sublat.data(),w,subpat.data());
        int swid=SIN*w, off=(ck.a-ck.lo)*SIN, ow=(ck.b-ck.a)*SIN, ocol=ck.a*SIN;
        for(int ch=0;ch<OUTCH;ch++)
            std::memcpy(out_patches+(size_t)ch*L+ocol, subpat.data()+(size_t)ch*swid+off, ow*sizeof(float));
    }
}
} // extern "C"
