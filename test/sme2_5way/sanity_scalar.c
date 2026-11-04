#include "math_core.h"
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int16_t g[8];
    uint64_t st=UINT64_C(0x701e2188);
    unsigned b,t;
    bw_gammas(g);
    for(b=0;b<8;b++) {
        if(bw_powmod(bw_mod(g[b]),8)!=3328) {
            fprintf(stderr,"gamma[%u] invalid: %d\n",b,g[b]);return 1;
        }
    }
    for(t=0;t<1000;t++) {
        int16_t a[256],c[256],i;
        for(i=0;i<256;i++)a[i]=bw_center((int)(bw_rand(&st)%BW_Q));
        memcpy(c,a,sizeof(a));
        bw_ntt3_scalar(c);bw_intt3(c);
        for(i=0;i<256;i++)if(bw_mod(a[i])!=bw_mod(c[i])){
            fprintf(stderr,"FAIL forward+inverse trial=%u i=%d\n",t,i);return 1;
        }
    }
    puts("PASS: 1000 scalar forward/inverse NTT3 trials; gamma^8=-1");
    return 0;
}
