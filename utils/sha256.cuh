//
// Created by mrhax on 9/23/24.
//

#ifndef SHA256_CUH
#define SHA256_CUH

#include <string>

__device__ void sha256(const char* input, size_t len, char* output);

#endif //SHA256_CUH
