//
// Created by mrhax on 9/23/24.
//

#include "Blockchain.cuh"
#include "Block.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "../utils/constants.hpp"

#define CUDA_CHECK(call)                                                         \
{                                                                                \
cudaError_t err = call;                                                      \
if (err != cudaSuccess) {                                                    \
std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
exit(EXIT_FAILURE);                                                      \
}                                                                            \
}


// Blockchain constructor
Blockchain::Blockchain(int difficulty) : difficulty(difficulty) {
    addBlock("Genesis Block");
}

// Add block to the blockchain
void Blockchain::addBlock( char* data) {
    Block newBlock(blockchain.size(), time(nullptr), data);

    //TODO put in a for loop to iter for more hashes and increase in steps of blockIdx.x * blockDim.x + threadIdx.x
    //It may be necessary to call another device just to get this variables? Idk

    uint32_t base_nonce = 0;
    uint32_t d_output[MINING_BLOCKS*MINING_BLOCK_THREADS]; //TODO change to a char[65] to contain the sha-256 hash
    hashKernel<<<MINING_BLOCKS, MINING_BLOCK_THREADS>>>(&newBlock, base_nonce, d_output);

    std::cout << "Found XOR hashes: " << std::endl;

    for (uint32_t hash : d_output) {
        std::cout << hash << std::endl;
    }

    blockchain.push_back(newBlock);
}
