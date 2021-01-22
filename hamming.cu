#include <cstdint>
#include <random>
#include <stdlib.h>
#include <time.h>

#define _POSIX_C_SOURCE 199309L
#define COUNT 5000

#ifdef __DRIVER_TYPES_H__
    #ifndef DEVICE_RESET
        #define DEVICE_RESET cudaDeviceReset();
    #endif
#else
    #ifndef DEVICE_RESET
        #define DEVICE_RESET 
    #endif
#endif

#define checkCudaErrors(val) { check((val), __LINE__); }

void check(cudaError_t cudaStatus, int line)
{
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d at line %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
        DEVICE_RESET
            exit(EXIT_FAILURE);
    }
}

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_int_distribution<unsigned long long> dis(
    std::numeric_limits<std::uint64_t>::min(),
    std::numeric_limits<std::uint64_t>::max()
);

void generateSeqs(uint64_t* seqs) 
{
    for (uint64_t i = 0; i < COUNT * 8; i++) {
        seqs[i] = (dis(gen) >> 63) & UINT64_MAX;
    }
}

void printSeqs(uint64_t* seqs) 
{
    for (long i = 0; i < COUNT * 8; i += 8)
    {
        printf("Sequence %li:  ", i / 8 + 1);
        for (long j = 0; j < 8; j++)
        {
            printf("%I64i ", seqs[i + j]);
        }
        printf("\n");
    }
}

//-------------------CPU SOLUTION------------------------

int hammingDistance(uint64_t n1, uint64_t n2)
{
    uint64_t x = n1 ^ n2;
    long setBits = 0;

    while (x > 0) {
        setBits += x & 1;
        x >>= 1;
    }

    return setBits;
}

void hamming(const uint64_t* seqs, bool* pairs) 
{
    long distance;

    for (long i = 0; i < COUNT; i++) 
    {
        for (long j = i + 1; j < COUNT; j++) 
        {
            distance = 0;

            for (long k = 0; k < 8; k++)
                distance += hammingDistance(seqs[i * 8 + k], seqs[j * 8 + k]);

            if (distance == 1)
                pairs[i * COUNT + j] = true;
        }
    }
}

//-------------------GPU SOLUTION------------------------

__device__ int hammingDistanceCuda(uint64_t n1, uint64_t n2)
{
    uint64_t x = n1 ^ n2;
    long setBits = 0;

    while (x > 0) 
    {
        setBits += x & 1;
        x >>= 1;
    }

    return setBits;
}

__global__ void isHammingOneCuda(const uint64_t* seqs, bool* pairs) 
{
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;

    unsigned int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    uint64_t comparedSeq[8];
    for (unsigned int i = 0; i < 8; i++) 
        comparedSeq[i] = seqs[globalThreadNum * 8 + i];

    for (unsigned int i = globalThreadNum + 1; i < COUNT; i++)
    {
        long distance = 0;

        for (unsigned int j = 0; j < 8; j++) 
            distance += hammingDistanceCuda(comparedSeq[j], seqs[i * 8 + j]);

        if (distance == 1) 
            pairs[globalThreadNum * COUNT + i] = true;
    }
}

cudaError_t hammingCuda(const uint64_t* seqs, bool* pairs) 
{
    dim3 block(32, 4);
    dim3 grid(block.x * block.y, ceil((double)COUNT / (block.x * block.y)));

    isHammingOneCuda <<<grid, block>>> (seqs, pairs);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return cudaSuccess;
}

//-------------------MAIN------------------------

int main() 
{
    uint64_t *seqs;
    bool *pairs;
    clock_t start, finish;
    double duration;
    uint64_t counter = 0;

    cudaMallocManaged(&seqs, COUNT * 8 * sizeof(uint64_t));
    cudaMallocManaged(&pairs, COUNT * COUNT * sizeof(bool));
    generateSeqs(seqs);
    printSeqs(seqs);

    for (long i = 0; i < COUNT * COUNT; i++)
        pairs[i] = false;

    printf("-------------CPU SOLUTION------------\n");
    start = clock();
    hamming(seqs, pairs);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Time:  %2.3f seconds\n", duration);
    
    for (uint64_t i = 0; i < COUNT * COUNT; i++)
        if (pairs[i])
            counter++;
    printf("Pairs with Hamming distance of 1:  %I64i\n", counter);
    
    counter = 0;
    for (long i = 0; i < COUNT * COUNT; i++)
        pairs[i] = false;

    printf("\n-------------GPU SOLUTION------------\n");
    start = clock();
    hammingCuda(seqs, pairs);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Time:  %2.3f seconds\n", duration);

    for (uint64_t i = 0; i < COUNT * COUNT; i++)
        if (pairs[i])
            counter++;
    printf("Pairs with Hamming distance of 1:  %I64i\n", counter);

    checkCudaErrors(cudaFree(seqs));
    checkCudaErrors(cudaFree(pairs));
	getchar();
    return 0;
}
