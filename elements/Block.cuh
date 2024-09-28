//
// Created by mrhax on 9/23/24.
//
#ifndef BLOCK_CUH
#define BLOCK_CUH
#include <cstdint>
#include "../utils/constants.hpp"


class Block {
public:
    uint32_t blockIndex;
    char previousBlockHash[65];
    time_t timeOfCreation;
    char data[MAX_DATA_SIZE];
    char currentHash[65];
    bool valid_nonce;
    uint64_t verified_nonce;

    Block(uint32_t block_index, time_t time_of_creation, char* inputData);

};

__global__ void hashKernel(char* device_input_data, uint32_t nonce_increment, uint32_t nonce_insert_index, char* output, uint32_t difficulty);
__device__ uint32_t performHash(uint32_t nonce, char* data);


extern __device__ uint32_t stop_flag;
extern __device__ uint32_t resulting_nonce;

#endif //BLOCK_CUH
