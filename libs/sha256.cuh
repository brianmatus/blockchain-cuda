#ifndef SHA256_CUH
#define SHA256_CUH

#include <cstdint>

__device__ void cuda_sha256_transform(uint32_t state[8], const uint8_t data[64]);
__device__ void cuda_sha256(uint8_t* data, int len, uint8_t* out);

#endif
