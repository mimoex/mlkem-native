#ifndef SME2_5WAY_NTT_BRIDGE_H
#define SME2_5WAY_NTT_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
size_t bench5_native_ctx_bytes(void);
int bench5_native_prepare(void *,const int16_t [4][3][256]);
int bench5_native_eval(const void *,const int16_t (*)[3][256],int16_t (*)[4][256],unsigned);
size_t bench5_portable_ctx_bytes(void);
int bench5_portable_prepare(void *,const int16_t [4][3][256]);
int bench5_portable_eval(const void *,const int16_t (*)[3][256],int16_t (*)[4][256],unsigned);
#endif
