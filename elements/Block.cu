#include "Block.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "../libs/sha256.hpp"

__device__ bool cudaIsValidNonce(const char* hashStr, int difficulty) {
    for (int i = 0; i < difficulty; ++i) {
        if (hashStr[i] != '0') return false;
    }
    return true;
}

__device__ void itoa(uint32_t num, char* buffer) {
    int i = 0;

    // Handle 0 explicitly, since it's a special case
    if (num == 0) {
        buffer[i++] = '0';
        buffer[i] = '\0';
        return;
    }

    // Process individual digits
    while (num > 0) {
        buffer[i++] = (num % 10) + '0'; // Get last digit
        num /= 10; // Remove last digit
    }

    buffer[i] = '\0'; // Null-terminate the string

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = buffer[j];
        buffer[j] = buffer[i - j - 1];
        buffer[i - j - 1] = temp;
    }
}

__device__ void formatHashString(char* hashStr, uint32_t index, unsigned long long time, const char* sData, uint32_t nonce, const char* prevHash) {
    int offset = 0;

    // Convert index to string
    char indexStr[12]; // Enough to hold 32-bit unsigned integer
    itoa(index, indexStr);
    while (indexStr[offset] != '\0') {
        hashStr[offset++] = indexStr[offset];
    }

    // Convert time to string
    char timeStr[21]; // Enough to hold unsigned long long
    unsigned long long t = time;
    int len = 0;
    do {
        timeStr[len++] = (t % 10) + '0';
        t /= 10;
    } while (t > 0);
    for (int j = len - 1; j >= 0; j--) {
        hashStr[offset++] = timeStr[j]; // Reverse the string
    }

    // Copy sData to hashStr
    while (*sData && offset < 64) {
        hashStr[offset++] = *sData++;
    }

    // Convert nonce to string
    char nonceStr[12]; // Enough to hold 32-bit unsigned integer
    itoa(nonce, nonceStr);
    while (nonceStr[offset] != '\0') {
        hashStr[offset++] = nonceStr[offset];
    }

    // Copy prevHash to hashStr
    while (*prevHash && offset < 64) {
        hashStr[offset++] = *prevHash++;
    }

    // Null-terminate the string
    hashStr[offset] = '\0';
}





__global__ void mineKernel(Block* block, int* foundNonce, int blockSize, int difficulty) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    char hashStr[65];
    char sData[512];
    char sPrevHash[65]; // Changed to accommodate null terminator

    for (int i = idx; i < blockSize && !(*foundNonce); i += stride) {
        // Manually copy _sData
        for (int j = 0; j < sizeof(sData) - 1 && block[i]._sData[j] != '\0'; ++j) {
            sData[j] = block[i]._sData[j];
        }
        sData[sizeof(sData) - 1] = '\0'; // Ensure null termination

        // Manually copy sPrevHash
        for (int j = 0; j < sizeof(sPrevHash) - 1 && block[i].sPrevHash[j] != '\0'; ++j) {
            sPrevHash[j] = block[i].sPrevHash[j];
        }
        sPrevHash[sizeof(sPrevHash) - 1] = '\0'; // Ensure null termination

        // Use the custom format function
        formatHashString(hashStr, block[i]._nIndex, block[i]._tTime, sData, block[i]._nNonce, sPrevHash);

        // Check if nonce is valid
        if (cudaIsValidNonce(hashStr, difficulty)) {
            *foundNonce = i;
            printf("Found hash %s\n", hashStr);
        }
    }
}


Block::Block(const uint32_t nIndexIn, char sDataIn[512]) : _nIndex(nIndexIn), _sData(sDataIn) {
    _nNonce = 0;
    _tTime = time(nullptr);
}

void Block::MineBlock(uint32_t difficulty) {
    int blockSize = 1;  // Adjust based on your GPU capability //256
    int threadSize = 1; // Adjust based on your GPU capability //1024
    int foundNonce = 0;

    // Launch the CUDA mining kernel
    launchCudaMine(this, blockSize, threadSize, difficulty);
}

void Block::launchCudaMine(Block* block, int blockSize, int threadSize, uint32_t difficulty) {
    int* d_foundNonce;
    cudaMalloc(&d_foundNonce, sizeof(int));
    cudaMemcpy(d_foundNonce, &block->_nNonce, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    mineKernel<<<blockSize, threadSize>>>(block, d_foundNonce, blockSize, difficulty);

    cudaMemcpy(&block->_nNonce, d_foundNonce, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_foundNonce);
}

char *Block::GetHash() {
    return _sHash.data();
}


std::string Block::calculateHashWithNonce(uint32_t nonce) const {
    std::stringstream ss;
    ss << _nIndex << _tTime << _sData << nonce << sPrevHash;
    return sha256(ss.str());
}