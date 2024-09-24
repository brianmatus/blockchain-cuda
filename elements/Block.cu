//
// Created by mrhax on 9/23/24.
//

#include "Block.cuh"
#include "../utils/sha256.cuh"
#include <cstring>


Block::Block(const uint32_t block_index, const time_t time_of_creation, char inputData[MAX_DATA_SIZE]) : blockIndex(block_index), timeOfCreation(time_of_creation) {
    verified_nonce = 1;
    valid_nonce = false;
    memset(previousBlockHash, 0, sizeof(previousBlockHash));
    memset(data, 0, sizeof(inputData));
    memset(currentHash, 0, sizeof(currentHash));


    //TODO change to memcopy to fill string buffer, random placerholder in the meantine
    data[0] = 'a';
    data[1] = 'b';
    data[2] = 'c';
    data[3] = 'd';


    // strcpy(previousBlockHash, "");
    // strcpy(data, "");
    // strcpy(currentHash, "");
}



__global__ void hashKernel(Block* block, uint32_t base_nonce, uint32_t* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = base_nonce + idx;
    uint32_t hash = performHash(nonce, block->data);
    output[idx] = hash;
}

//TODO change to sha256 later
__device__ uint32_t performHash(uint32_t nonce, char* data) {
    uint32_t hash = nonce;
    for (int i = 0; i < MAX_DATA_SIZE; ++i) {
        hash ^= data[i];
    }
    return hash;

}