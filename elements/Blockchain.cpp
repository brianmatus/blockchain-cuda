//
// Created by mrhax on 9/22/24.
//

#include "Blockchain.hpp"

#include <cstring>

#include "Block.cuh"


Blockchain::Blockchain() {
    char genesis[512] = "Genesis block";
    _vChain.emplace_back(0, genesis);
    _nDifficulty = 5;
}


void Blockchain::AddBlock(Block bNew) {
    strncpy(bNew.sPrevHash, _GetLastBlock().GetHash(),65);
    bNew.MineBlock(_nDifficulty);
    _vChain.push_back(bNew);
}

Block Blockchain::_GetLastBlock() const {
    return _vChain.back();
}

