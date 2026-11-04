#include <stdio.h>
#include <stdlib.h>
#include "partial_scalar.h"
int main(void){
    int16_t (*pub)[3][256]=malloc(sizeof(int16_t[4][3][256]));
    int16_t (*sec)[3][256]=malloc(sizeof(int16_t[16][3][256]));
    int16_t (*out)[4][256]=malloc(sizeof(int16_t[16][4][256]));
    bw_scalar_ctx *ctx=malloc(sizeof(*ctx));
    unsigned t,r,d,i;
    if(!pub||!sec||!out||!ctx)return 2;
    for(t=1;t<=12;t++){
        bw_fill_input(pub,sec,16,t);
        if(bw_scalar_prepare(ctx,pub)||bw_scalar_eval(ctx,sec,out,16))return 3;
        for(r=0;r<16;r++){
            int16_t oracle[4][256];
            bw_oracle_one(pub,sec[r],oracle);
            for(d=0;d<4;d++)for(i=0;i<256;i++)if(bw_mod(out[r][d][i])!=bw_mod(oracle[d][i])) {
                fprintf(stderr,"FAIL trial=%u request=%u d=%u i=%u actual=%d expected=%d\n",t,r,d,i,out[r][d][i],oracle[d][i]);return 1;
            }
        }
    }
    puts("PASS: partial scalar agrees with direct negacyclic convolution (192 requests x 4 polys x 256 coeffs)");
    return 0;
}
