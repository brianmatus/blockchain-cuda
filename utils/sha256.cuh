//
// Created by mrhax on 9/23/24.
//

#ifndef SHA256_CUH
#define SHA256_CUH

class SHA256 {
protected:
    typedef unsigned char uint8;
    typedef unsigned int uint32;
    typedef unsigned long long uint64;

    static const unsigned int SHA224_256_BLOCK_SIZE = 64; //512/8
public:
    __device__ SHA256() {};
    __device__ void init();
    void __device__ update(const unsigned char *message, unsigned int len);
    void __device__ final(unsigned char *digest);
    static __const__ unsigned int DIGEST_SIZE = 32; // 256/8

protected:
    __device__ void transform(const unsigned char *message, unsigned int block_nb);
    unsigned int m_tot_len;
    unsigned int m_len;
    unsigned char m_block[2*SHA224_256_BLOCK_SIZE];
    uint32 m_h[8];
};

__device__ void sha256(const char* input, unsigned int input_length, char* output);

#define SHA2_SHFR(x, n)    (x >> n)
#define SHA2_ROTR(x, n)   ((x >> n) | (x << ((sizeof(x) << 3) - n)))
#define SHA2_ROTL(x, n)   ((x << n) | (x >> ((sizeof(x) << 3) - n)))
#define SHA2_CH(x, y, z)  ((x & y) ^ (~x & z))
#define SHA2_MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA256_F1(x) (SHA2_ROTR(x,  2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22))
#define SHA256_F2(x) (SHA2_ROTR(x,  6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25))
#define SHA256_F3(x) (SHA2_ROTR(x,  7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x,  3))
#define SHA256_F4(x) (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10))
#define SHA2_UNPACK32(x, str)                 \
{                                             \
*((str) + 3) = (uint8) ((x)      );       \
*((str) + 2) = (uint8) ((x) >>  8);       \
*((str) + 1) = (uint8) ((x) >> 16);       \
*((str) + 0) = (uint8) ((x) >> 24);       \
}
#define SHA2_PACK32(str, x)                   \
{                                             \
*(x) =   ((uint32) *((str) + 3)      )    \
| ((uint32) *((str) + 2) <<  8)    \
| ((uint32) *((str) + 1) << 16)    \
| ((uint32) *((str) + 0) << 24);   \
}

#endif //SHA256_CUH
