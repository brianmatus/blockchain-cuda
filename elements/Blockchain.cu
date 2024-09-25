//
// Created by mrhax on 9/23/24.
//

#include "Blockchain.cuh"
#include "Block.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "../utils/constants.hpp"


// Blockchain constructor
Blockchain::Blockchain(int difficulty) : difficulty(difficulty) {
    //TODO set difficulty to 0 or 1 just to insert genesis block? or export Blockchain.setDifficulty?
    addBlock("Genesis Block");
}

// Add block to the blockchain
void Blockchain::addBlock( const std::string& data) {
    //TODO put in a for loop to iter for more hashes
    // and increase in steps of blockIdx.x * blockDim.x + threadIdx.x (=MINING_TOTAL_THREADS)
    uint32_t base_nonce = 0;

    char dataArr[MAX_DATA_SIZE] = {};
    memcpy(dataArr, data.c_str(), MAX_DATA_SIZE);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///Block
    Block h_block(blockchain.size(), time(nullptr), dataArr);
    Block* d_block;
    cudaMalloc(&d_block, sizeof(Block));
    cudaMemcpy(d_block, &h_block, sizeof(Block), cudaMemcpyHostToDevice);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///Output
    uint32_t h_output[MINING_TOTAL_THREADS] = {}; //TODO change to a char[65] to contain the sha-256 hash

    uint32_t* d_output;
    cudaMalloc(&d_output, sizeof(uint32_t) * MINING_TOTAL_THREADS);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    hashKernel<<<MINING_SM_BLOCKS, MINING_BLOCK_THREADS>>>(d_block, base_nonce, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(uint32_t)*MINING_TOTAL_THREADS, cudaMemcpyDeviceToHost);

    std::cout << "Found XOR hashes: " << std::endl;
    for (const uint32_t hash : h_output) {
        std::cout << hash << std::endl;
    }

    cudaFree(d_output);
    blockchain.push_back(h_block);
}
