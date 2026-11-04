#include <stdio.h>
#include <stdlib.h>
#include "direct_core.h"
int main(void){
    int16_t (*pub)[3][256]=malloc(sizeof(int16_t[4][3][256]));
    int16_t (*sec)[3][256]=malloc(sizeof(int16_t[16][3][256]));
    int16_t (*pack)[128][32]=malloc(sizeof(int16_t[3][128][32]));
    int16_t (*out)[4][256]=malloc(sizeof(int16_t[16][4][256]));
    int32_t (*raw)[4][2][256]=malloc(sizeof(int32_t[8][4][2][256]));
    bw_direct_ctx *ctx=malloc(sizeof(*ctx));
    unsigned t,r,d,i;
    if(!pub||!sec||!pack||!out||!raw||!ctx)return 2;
    for(t=1;t<=3;t++){
      bw_fill_input(pub,sec,16,t);
      if(bw_direct_prepare(ctx,pub))return 3;
      bw_direct_pack(sec,pack);
      bw_direct_emulate(ctx,pack,raw);
      bw_direct_scatter(raw,out);
      for(r=0;r<16;r++){
        int16_t oracle[4][256];
        bw_oracle_one(pub,sec[r],oracle);
        for(d=0;d<4;d++)for(i=0;i<256;i++)if(bw_mod(out[r][d][i])!=bw_mod(oracle[d][i])) {
            fprintf(stderr,"FAIL t=%u r=%u d=%u i=%u actual=%d expected=%d\n",t,r,d,i,out[r][d][i],oracle[d][i]);return 1;
        }
      }
    }
    puts("PASS: Direct-SMOPA matrix layout/emulator matches 256x256 negacyclic reference (48 requests)");
    return 0;
}
