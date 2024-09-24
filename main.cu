#include "elements/Blockchain.cuh"


int main() {
    // Initialize blockchain with difficulty 4
    Blockchain blockchain(4);

    blockchain.addBlock({});
    blockchain.addBlock({});

    return 0;
}





#include <cuda_runtime.h>
#include <iostream>

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
