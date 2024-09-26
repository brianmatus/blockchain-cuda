//
// Created by mrhax on 9/23/24.
//

#include "Block.cuh"
#include "../utils/sha256.cuh"
#include <cstring>

__device__ int stop_flag = 0;

Block::Block(const uint32_t block_index, const time_t time_of_creation, char* inputData) : blockIndex(block_index), timeOfCreation(time_of_creation) {
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


__global__ void hashKernel(char* device_input_data, uint32_t nonce_increment, uint32_t nonce_insert_index, char* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = nonce_increment + idx;

    //TODO change base_nonce to a nonce_increment=MINING_TOTAL_THREADS
    //TODO while not found and global flag is false:
    // Keep going in increments of nonce_increment. Set global flag if found

    //TODO uncomment
    insert_nonce(device_input_data, nonce, nonce_insert_index);

    char resulting_hash[65] = {};
    sha256(device_input_data, nonce_insert_index + 4, resulting_hash); // nonce_insert_index + 4 because of nonce

    for (int i = 0; i < 65; ++i) {
        output[i] = resulting_hash[i];
    }
}

//TODO change to sha256 later
__device__ uint32_t performHash(uint32_t nonce, char* data) {
    uint32_t hash = nonce;
    for (int i = 0; i < MAX_DATA_SIZE; ++i) {
        const char a = data[i];
        if (a == '\0') break;
        hash ^= a;
        // hash = i;
    }
    return hash;
}

__device__ int unsigned_long_to_str(unsigned long value, char* buffer, int offset) {
    char temp[20];  // Enough to hold the largest 64-bit unsigned long (20 digits)
    int length = 0;

    // Special case for value 0
    if (value == 0) {
        buffer[offset] = '0';
        return 1;  // Returning length of 1 since '0' is a single character
    }

    // Convert the number to a string by extracting digits from least to most significant
    while (value > 0) {
        temp[length++] = '0' + (value % 10);  // Get the last digit as a character
        value /= 10;  // Move to the next digit
    }

    // Reverse the order of digits and write them to the buffer at the given offset
    for (int i = 0; i < length; i++) {
        buffer[offset + i] = temp[length - 1 - i];
    }

    // Return the number of characters written
    return length;
}
