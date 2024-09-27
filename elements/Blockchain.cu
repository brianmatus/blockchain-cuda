//
// Created by mrhax on 9/23/24.
//

#include "Blockchain.cuh"
#include "Block.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "../utils/constants.hpp"
#include <sstream>
#include "../utils/sha256.cuh"


// Blockchain constructor
Blockchain::Blockchain(int difficulty) : difficulty(difficulty) {
    const std::string s = "Genesis Block";
    char dataArr[MAX_DATA_SIZE] = {};
    memcpy(dataArr, s.c_str(), s.length());

    Block genesis_block(blockchain.size(), time(nullptr), dataArr);
    genesis_block.currentHash[0] = '0';
    blockchain.push_back(genesis_block);
}

// Add block to the blockchain
void Blockchain::addBlock( const std::string& data) {
    // Reset vars before launching the next kernel
    uint32_t h_stop_flag = 0;
    cudaMemcpyToSymbol(stop_flag, &h_stop_flag, sizeof(uint32_t));

    uint32_t h_resulting_nonce = 0;
    cudaMemcpyToSymbol(resulting_nonce, &h_resulting_nonce, sizeof(uint32_t));

    char dataArr[MAX_DATA_SIZE] = {};
    memcpy(dataArr, data.c_str(), data.length());

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///Block
    Block h_block(blockchain.size(), time(nullptr), dataArr);
    memcpy(h_block.previousBlockHash, blockchain.back().currentHash, 64); //TODO is already null terminated?

    std::stringstream ss;
    ss << h_block.blockIndex  << "\n" << h_block.timeOfCreation<< "\n" << h_block.previousBlockHash<< "\n" << dataArr << "\n";
    const std::string resulting = ss.str();

    std::cout << "Resulting block: " << std::endl;
    std::cout << resulting << std::endl;

    char* d_block_data;
    cudaMalloc(&d_block_data, MAX_DATA_SIZE);
    cudaMemcpy(d_block_data, resulting.c_str(), resulting.length(), cudaMemcpyHostToDevice);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///Output
    char h_output[65] = {}; //TODO change to a char[65] to contain the sha-256 hash

    char* d_output;
    cudaMalloc(&d_output, sizeof(char) * 65);
    cudaMemset(d_output, 0, sizeof(char) * 65);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    hashKernel<<<MINING_SM_BLOCKS, MINING_BLOCK_THREADS>>>(d_block_data, MINING_TOTAL_THREADS, resulting.length(), d_output, difficulty);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(h_output, d_output, sizeof(char)*65, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&h_resulting_nonce, resulting_nonce, sizeof(uint32_t));

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Resulting hash" << std::endl;
    std::cout << h_output << std::endl;
    std::cout << "For nonce:" << std::endl;
    std::cout << h_resulting_nonce << std::endl;

    cudaFree(d_output);
    blockchain.push_back(h_block);
}
