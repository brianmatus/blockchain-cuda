//
// Created by mrhax on 9/23/24.
//

#include "Block.cuh"

#include <cstdio>

#include "../utils/sha256.cuh"
#include <cstring>

__device__ uint32_t stop_flag = 0;
__device__ uint32_t resulting_nonce = 0;

Block::Block(const uint32_t block_index, const time_t time_of_creation, const char* inputData) : blockIndex(block_index), timeOfCreation(time_of_creation) {
    verified_nonce = 0;
    valid_nonce = false;
    memset(previousBlockHash, 0, sizeof(previousBlockHash));
    memset(currentHash, 0, sizeof(currentHash));
    memset(data, 0, sizeof(inputData));
    memcpy(data, inputData, sizeof(inputData));

    // strcpy(previousBlockHash, "");
    // strcpy(data, "");
    // strcpy(currentHash, "");
}

__device__ void insert_nonce(char* device_input_data, uint32_t nonce, uint32_t nonce_insert_index) {
    // Break down the nonce into 4 bytes, little-endian
    device_input_data[nonce_insert_index]     = (char)(nonce & 0xFF);         // Least significant byte
    device_input_data[nonce_insert_index + 1] = (char)((nonce >> 8) & 0xFF);  // Next byte
    device_input_data[nonce_insert_index + 2] = (char)((nonce >> 16) & 0xFF); // Next byte
    device_input_data[nonce_insert_index + 3] = (char)((nonce >> 24) & 0xFF); // Most significant byte
}


__device__ bool check_leading_zeros(const char* hash, uint32_t num_zeros) {
    for (int i = 0; i < num_zeros; i++) {
        if (hash[i] != '0') {
            return false;
        }
    }
    return true;
}


__global__ void hashKernel(const char* device_input_data, const uint32_t nonce_increment, const uint32_t nonce_insert_index, char* output, const uint32_t difficulty) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = 0;
    uint32_t nonce_base = 0;

    char local_data[MAX_DATA_SIZE];
    for (int i = 0; i < MAX_DATA_SIZE; ++i) {
        local_data[i] = device_input_data[i];
    }

    while (atomicCAS(&stop_flag, 0, 0) == 0) {
        nonce = nonce_base + idx;
        insert_nonce(local_data, nonce, nonce_insert_index);

        char resulting_hash[65] = {};
        sha256(local_data, nonce_insert_index + 4, resulting_hash); // nonce_insert_index + 4 because of nonce of size uint32_t

        // printf("nonce:%i, resulting_hash:%s\n", nonce, resulting_hash);

        if (check_leading_zeros(resulting_hash, difficulty)) {
            if (atomicCAS(&stop_flag, 0, 1) == 0) {
                printf("----------------------------------------------\nBlock successfully mined by idx:%i nonce:%i\n%s\n", idx, nonce, local_data);
                for (int i = 0; i < 65; ++i) {
                    output[i] = resulting_hash[i];
                }
                atomicExch(&resulting_nonce, nonce);
            }
            break;
        }
        nonce_base += nonce_increment;
    }
}

__device__ uint32_t performHash(const uint32_t nonce, const char* data) {
    uint32_t hash = nonce;
    for (int i = 0; i < MAX_DATA_SIZE; ++i) {
        const char a = data[i];
        if (a == '\0') break;
        hash ^= a;
    }
    return hash;
}
