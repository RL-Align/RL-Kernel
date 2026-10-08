#include "bitmath.h"
#include <stddef.h>
#include <fenv.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
extern "C" int p3_host_environment() {
    if (fegetround()!=FE_TONEAREST) return 0;
#if defined(__SSE__)
    if (_mm_getcsr() & 0x8040u) return 0; // FTZ or DAZ would silently change subnormals.
#endif
    return 1;
}
extern "C" void p3_math(const float* in,float* out,size_t n,int op) {
    for (size_t i=0;i<n;++i) {
        float x=in[i];
        switch(op) {
            case 0: out[i]=p3::exp(x); break;
            case 1: out[i]=p3::log1p(x); break;
            case 2: out[i]=p3::softplus(x); break;
            case 3: out[i]=p3::sigmoid(x); break;
            case 4: out[i]=p3::root(x); break;
            case 5: out[i]=p3::bf16(x); break;
        }
    }
}
