#include <iostream>

#include "elements/Blockchain.hpp"

int main() {
    auto bChain = Blockchain();
    cout << "Mining block 1..." << endl;
    char data1[512] = "Block 1 Data";
    Block b1 = Block(1, data1);
    bChain.AddBlock(b1);
    //
    // cout << "Mining block 2..." << endl;
    // bChain.AddBlock(Block(2, "Block 2 Data"));
    //
    // cout << "Mining block 3..." << endl;
    // bChain.AddBlock(Block(3, "Block 3 Data"));


    return 0;
}


// #include <cuda_runtime.h>
//
// int main() {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0); // Assuming device 0 (first GPU)
//
//     std::cout << "GPU: " << prop.name << std::endl;
//     std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
//     std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
//     std::cout << "Max Threads Dim (Block Dimension): [" << prop.maxThreadsDim[0] << ", "
//               << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
//     std::cout << "Max Grid Size: [" << prop.maxGridSize[0] << ", "
//               << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
//     std::cout << "Warp Size: " << prop.warpSize << std::endl;
//     std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
//
//
//     // int blockSize;   // The launch configurator returned block size
//     // int minGridSize; // The minimum grid size needed to achieve the maximum occupancy
//     // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
//
//     return 0;
// }