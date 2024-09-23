#ifndef BLOCK_H
#define BLOCK_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

class Block {
public:
    char sPrevHash[65];
    Block(uint32_t nIndexIn, char sDataIn[512]);
    char* GetHash();

    uint32_t _nNonce;
    std::string hash;
    void MineBlock(uint32_t difficulty);
    bool isValidNonce(int nonce) const;
    std::string calculateHashWithNonce(uint32_t nonce) const;

    uint32_t _nIndex;
    char *_sData;
    std::string _sHash;
    time_t _tTime;
    
private:
    std::string calculateHash(int nonce) const;
    static void launchCudaMine(Block* block, int blockSize, int threadSize, uint32_t difficulty);
};

#endif
