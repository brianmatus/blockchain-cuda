//
// Created by mrhax on 9/23/24.
//

#ifndef BLOCKCHAIN_HPP
#define BLOCKCHAIN_HPP

#include <string>
#include <vector>
#include "Block.cuh"

class Blockchain {
public:
    Blockchain(int difficulty);
    void addBlock( char *data);

private:
    std::vector<Block> blockchain;
    int difficulty;
    void mineBlockGPU(int blockIdx);
};

#endif //BLOCKCHAIN_HPP
