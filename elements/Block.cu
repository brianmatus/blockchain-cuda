#include "Block.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "../libs/sha256.cuh"

// Constructor definition
Block::Block(uint32_t nIndexIn, char sDataIn[512]) {
    _nIndex = nIndexIn;
    _sData = new char[512];
    strncpy(_sData, sDataIn, 512);
    _nNonce = 0;
    _tTime = time(nullptr);  // Record the block creation time
}

// GetHash function
char* Block::GetHash() {
    return const_cast<char*>(_sHash.c_str());
}

// The mining function
void Block::MineBlock(uint32_t difficulty) {
    // For simplicity, using the CPU for now, but launchCudaMine would be called for the GPU version
    char* d_nonce;
    launchCudaMine(this, 1, 1, difficulty);

    // Print the resulting nonce for verification
    // std::cout << "Block mined: " << _sHash << std::endl;
}

// Helper function to check if a nonce is valid for the given difficulty
__device__ bool cudaIsValidNonce(const char* hashStr, int difficulty) {
    for (int i = 0; i < difficulty; ++i) {
        if (hashStr[i] != '0') return false;
    }
    return true;
}

// Helper function to convert an integer to a string (itoa equivalent)
__device__ void itoa(uint32_t num, char* buffer) {
    int i = 0;

    if (num == 0) {
        buffer[i++] = '0';
        buffer[i] = '\0';
        return;
    }

    while (num > 0) {
        buffer[i++] = (num % 10) + '0';
        num /= 10;
    }

    buffer[i] = '\0';

    for (int j = 0; j < i / 2; j++) {
        char temp = buffer[j];
        buffer[j] = buffer[i - j - 1];
        buffer[i - j - 1] = temp;
    }
}

// Kernel function for SHA-256 mining on the GPU
__global__ void sha256_kernel(Block* block, uint32_t start_nonce, uint32_t end_nonce, char* out_nonce, int difficulty) {
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check to ensure the nonce is within the expected range
    if (nonce >= end_nonce) return;

    char temp_hash[65];  // Local memory on the GPU

    // Iterate through nonces and check for a valid one
    while (nonce < end_nonce) {
        block->calculateHashWithNonce(nonce, temp_hash);

        if (cudaIsValidNonce(temp_hash, difficulty)) {
            // Copy the valid nonce to the output buffer, with bounds checking
            for (int i = 0; i < 64; i++) {
                if (i >= 0 && i < 65) {  // Ensure we're within bounds
                    out_nonce[i] = temp_hash[i];
                }
            }
            return;
        }
        nonce++;
    }
}


// Calculate hash with the given nonce using CUDA-compatible SHA-256
__device__ void Block::calculateHashWithNonce(uint32_t nonce, char* hashOut) const {
    // Prepare data for hashing (concatenate index, data, time, and nonce)
    char nonce_str[21];
    itoa(nonce, nonce_str);

    // Combine the block data and nonce into a single string
    std::string input = std::to_string(_nIndex) + _sData + std::to_string(_tTime) + nonce_str;

    // Calculate SHA-256 on the device
    uint8_t hash[32];  // 32 bytes for SHA-256
    cuda_sha256((uint8_t*)input.c_str(), input.length(), hash);

    // Convert hash to a hex string
    for (int i = 0; i < 32; i++) {
        sprintf(hashOut + (i * 2), "%02x", hash[i]);  // Convert binary hash to hex string
    }
    hashOut[64] = '\0'; // Ensure null-termination of the hex string
}


// Launch GPU mining
void Block::launchCudaMine(Block* block, int blockSize, int threadSize, uint32_t difficulty) {
    char* d_nonce;
    cudaError_t cudaStatus;

    // Allocate memory for nonce on the device
    cudaStatus = cudaMalloc(&d_nonce, 65 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_nonce!" << std::endl;
        return;
    }

    // Launch the mining kernel
    sha256_kernel<<<blockSize, threadSize>>>(block, 0, UINT32_MAX, d_nonce, difficulty);

    // Synchronize the device to ensure the kernel has finished executing
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_nonce);
        return;
    }

    // Copy the result back to the host
    char result_nonce[65];
    cudaStatus = cudaMemcpy(result_nonce, d_nonce, 65 * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_nonce);
        return;
    }

    // Ensure null-termination of the result_nonce string
    result_nonce[64] = '\0';  // Null-terminate the result_nonce

    // Print each character as a raw integer (ASCII value)
    std::cout << "Raw integer values of result_nonce: ";
    for (int i = 0; i < 65; i++) {
        std::cout << static_cast<int>(result_nonce[i]) << " ";
    }
    std::cout << std::endl;

    // Store the resulting hash in the block
    block->_sHash = std::string(result_nonce);

    std::cout << "Resulting Nonce: " << result_nonce << std::endl;
    std::cout << "Block mined: " << block->_sHash << std::endl;

    // Free GPU memory
    cudaFree(d_nonce);
}


