// SPDX-License-Identifier: Apache-2.0
// p3-bitmath.v1: shared host/device source. No exp/log libm or fused arithmetic.
#pragma once
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define P3_HD __host__ __device__
#else
#define P3_HD
#endif
namespace p3 {
P3_HD inline uint32_t bits(float x) { union { float f; uint32_t u; } v; v.f=x; return v.u; }
P3_HD inline float from_bits(uint32_t x) { union { float f; uint32_t u; } v; v.u=x; return v.f; }
P3_HD inline float add(float a,float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a,b);
#else
    volatile float r=a+b; return r;
#endif
}
P3_HD inline float mul(float a,float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a,b);
#else
    volatile float r=a*b; return r;
#endif
}
P3_HD inline float div(float a,float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a,b);
#else
    volatile float r=a/b; return r;
#endif
}
P3_HD inline float root(float x) {
#ifdef __CUDA_ARCH__
    return __fsqrt_rn(x);
#else
    return sqrtf(x); // IEEE correctly rounded sqrt, not reciprocal approximation.
#endif
}
P3_HD inline bool finite(float x) { return (bits(x)&0x7f800000u)!=0x7f800000u; }
P3_HD inline float bf16(float x) {
    uint32_t u=bits(x);
    if ((u&0x7f800000u)==0x7f800000u) return x;
    return from_bits((u+0x7fffu+((u>>16)&1u))&0xffff0000u);
}
P3_HD inline float exp(float x) {
    if (!finite(x)) return x < 0 ? 0.0f : x;
    if (x < -104.0f) return 0.0f;
    if (x > 88.72283935546875f) return from_bits(0x7f800000u);
    float y=mul(x,1.4426950408889634f);
    int n=(int)add(y,y>=0 ? 0.5f : -0.5f);
    float r=add(add(x,-mul((float)n,0.693145751953125f)),
                -mul((float)n,1.428606765330187e-6f));
    float p=1.0f/3628800.0f;
    p=add(1.0f/362880.0f,mul(r,p));
    p=add(1.0f/40320.0f,mul(r,p));
    p=add(1.0f/5040.0f,mul(r,p));
    p=add(1.0f/720.0f,mul(r,p));
    p=add(1.0f/120.0f,mul(r,p));
    p=add(1.0f/24.0f,mul(r,p));
    p=add(1.0f/6.0f,mul(r,p));
    p=add(0.5f,mul(r,p)); p=add(1.0f,mul(r,p)); p=add(1.0f,mul(r,p));
    if (n < -126) return mul(mul(p,from_bits(1u<<23)),from_bits((uint32_t)(n+126+127)<<23));
    if (n > 127) return mul(mul(p,from_bits(254u<<23)),2.0f);
    return mul(p,from_bits((uint32_t)(n+127)<<23));
}
P3_HD inline float log1p(float x) {
    if (!finite(x)) return x>0 ? x : from_bits(0x7fc00000u);
    if (x <= -1.0f) return from_bits(x==-1.0f ? 0xff800000u : 0x7fc00000u);
    if (x > -0.5f && x < 0.5f) {
        // log1p(x)=2*atanh(x/(2+x)); no 1+x cancellation, even for tiny exp values.
        if (x==0.0f) return x;
        if (x > -0.0001f && x < 0.0001f) {
            float p=add(-0.25f,mul(x,0.2f));
            p=add(1.0f/3.0f,mul(x,p)); p=add(-0.5f,mul(x,p));
            return mul(x,add(1.0f,mul(x,p)));
        }
        float v=div(x,add(2.0f,x)), v2=mul(v,v), p=1.0f/21.0f;
        for (int i=19;i>=1;i-=2) p=add(div(1.0f,(float)i),mul(v2,p));
        return mul(mul(2.0f,v),p);
    }
    float a=add(1.0f,x);
    uint32_t u=bits(a); int n=(int)(u>>23)-127;
    float m=from_bits((u&0x7fffffu)|(127u<<23));
    float v=div(add(m,-1.0f),add(m,1.0f)), v2=mul(v,v);
    float p=1.0f/21.0f;
    for (int i=19;i>=1;i-=2) p=add(div(1.0f,(float)i),mul(v2,p));
    return add(mul((float)n,0.6931471805599453f),mul(mul(2.0f,v),p));
}
P3_HD inline float softplus(float x) { return x>20.0f ? x : log1p(exp(x)); }
P3_HD inline float sigmoid(float x) {
    float t=exp(x>=0 ? -x : x);
    return x>=0 ? div(1.0f,add(1.0f,t)) : div(t,add(1.0f,t));
}
P3_HD inline float sum6(const float* a) {
    return add(add(add(a[0],a[1]),add(a[2],a[3])),add(a[4],a[5]));
}
} // namespace p3
