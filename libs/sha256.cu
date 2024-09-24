#include "sha256.cuh"

// Constants for SHA-256
__device__ const uint32_t d_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3
};

// Bit manipulation macros for SHA-256
__device__ __inline__ uint32_t ROTR(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __inline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __inline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __inline__ uint32_t Sigma0(uint32_t x) {
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__device__ __inline__ uint32_t Sigma1(uint32_t x) {
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

__device__ __inline__ uint32_t sigma0(uint32_t x) {
    return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3);
}

__device__ __inline__ uint32_t sigma1(uint32_t x) {
    return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10);
}

// SHA-256 transformation function
__device__ void cuda_sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t m[64];
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Load the message into the first 16 words w[0..15] of the array
    for (int i = 0; i < 16; ++i) {
        m[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | data[i * 4 + 3];
        w[i] = m[i];
    }

    // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array
    for (int i = 16; i < 64; ++i) {
        w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables to the current hash value
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // Compression function main loop
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + d_sha256_k[i] + w[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Add the compressed chunk to the current hash value
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA-256 hashing function
__device__ void cuda_sha256(uint8_t* data, int len, uint8_t* out) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t block[64];
    int block_count = len / 64;

    // Process each 512-bit block
    for (int i = 0; i < block_count; ++i) {
        memcpy(block, data + i * 64, 64);
        cuda_sha256_transform(state, block);
    }

    // Final block processing
    int rem_len = len % 64;
    memset(block, 0, 64);
    memcpy(block, data + block_count * 64, rem_len);
    block[rem_len] = 0x80;

    if (rem_len >= 56) {
        cuda_sha256_transform(state, block);
        memset(block, 0, 64);
    }

    uint64_t total_len_bits = len * 8;
    block[63] = total_len_bits & 0xFF;
    block[62] = (total_len_bits >> 8) & 0xFF;
    block[61] = (total_len_bits >> 16) & 0xFF;
    block[60] = (total_len_bits >> 24) & 0xFF;
    block[59] = (total_len_bits >> 32) & 0xFF;
    block[58] = (total_len_bits >> 40) & 0xFF;
    block[57] = (total_len_bits >> 48) & 0xFF;
    block[56] = (total_len_bits >> 56) & 0xFF;

    cuda_sha256_transform(state, block);

    // Copy the final state to the output
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = (state[i] >> 24) & 0xFF;
        out[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        out[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        out[i * 4 + 3] = state[i] & 0xFF;
    }
}
