//
// Created by mrhax on 9/24/24.
//

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cstdint>

constexpr uint32_t MAX_DATA_SIZE = 512;

constexpr uint32_t MINING_BLOCK_THREADS = 1024;
constexpr uint32_t MINING_BLOCKS = 192; //RTX 2080 Super has 48 SMs. Ideal is 2 (or more) blocks per SM. 48*4 = 192 //TODO increase later

#endif //CONSTANTS_HPP
