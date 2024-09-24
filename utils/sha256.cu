//
// Created by mrhax on 9/23/24.
//

#include <cstdint>
#include "sha256.cuh"
#include <cstring>
#include <cstdio>

#define ROTR(x,n) ((x >> n) | (x << (32 - n)))

// Constants defined in the SHA-256 standard
__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Initial hash values for SHA-256
__device__ __constant__ uint32_t h[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Padding function
__device__ void sha256_pad(uint8_t* padded, const char* input, size_t len) {
    int i = 0;
    while (i < len) {
        padded[i] = input[i];
        i++;
    }
    padded[len] = 0x80;  // Append '1' bit to the message
    len++;

    // Pad with zeros
    while ((len % 64) != 56) {
        padded[len++] = 0;
    }

    uint64_t bit_len = len * 8;  // Message length in bits
    for (int i = 0; i < 8; i++) {
        padded[len++] = (bit_len >> (56 - 8 * i)) & 0xFF;  // Append the original length of the message
    }
}

// SHA-256 compression function
__device__ void sha256_transform(uint32_t* state, const uint8_t* block) {
    uint32_t a, b, c, d, e, f, g, h1;
    uint32_t w[64];
    uint32_t t1, t2;

    for (int i = 0; i < 16; i++) {
        w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    }

    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i - 15], 7) ^ ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROTR(w[i - 2], 17) ^ ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h1 = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h1 + S1 + ch + k[i] + w[i];
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h1 = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h1;
}

// Helper function to convert a byte to a hexadecimal string (in device code)
__device__ void byteToHex(uint8_t byte, char* hex) {
    const char* hexChars = "0123456789abcdef";
    hex[0] = hexChars[(byte >> 4) & 0x0F];  // Extract upper 4 bits
    hex[1] = hexChars[byte & 0x0F];         // Extract lower 4 bits
}

// Convert hash result to hexadecimal string
__device__ void toHexString(uint32_t* hash, char* output) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            byteToHex((hash[i] >> (24 - j * 8)) & 0xFF, &output[i * 8 + j * 2]);
        }
    }
    output[64] = '\0';  // Null-terminate the string
}

// SHA-256 function on GPU
__device__ void sha256(const char* input, size_t len, char* output) {
    uint8_t padded[64 * 2] = {0};  // Padding buffer for two blocks of 512 bits
    sha256_pad(padded, input, len);

    // Initial hash values
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = h[i];
    }

    // Process each block of 512 bits
    for (int i = 0; i < len + 9; i += 64) {
        sha256_transform(state, padded + i);
    }

    // Convert the resulting hash into a hex string
    toHexString(state, output);
}
