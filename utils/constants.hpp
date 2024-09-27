//
// Created by mrhax on 9/24/24.
//

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cstdint>

constexpr uint32_t MAX_DATA_SIZE = 512;

constexpr uint32_t MINING_BLOCK_THREADS = 256; //1024
constexpr uint32_t MINING_SM_BLOCKS = 4; //RTX 2080 Super has 48 SMs. Ideal is 2 (or more) blocks per SM. 48*4 = 192 //TODO increase later
constexpr uint32_t MINING_TOTAL_THREADS = MINING_BLOCK_THREADS * MINING_SM_BLOCKS;

#endif //CONSTANTS_HPP
