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

    Block(const uint32_t block_index, const time_t time_of_creation, char inputData[MAX_DATA_SIZE]);

};

__global__ void hashKernel(Block* block, uint32_t base_nonce, uint32_t* output); //TODO change output to a char[65] for SHA-256
__device__ uint32_t performHash(uint32_t nonce, char* data);

#endif //BLOCK_CUH
