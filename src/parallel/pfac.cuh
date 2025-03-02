#ifndef PFAC_CUH
#define PFAC_CUH

#include "../include/common.h"
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants for CUDA
#define BLOCK_SIZE 256
#define MAX_GRID_SIZE 65535

// GPU data structure
typedef struct {
    int *d_trie;           // Transition table
    int *d_output;         // Output table for patterns
    char *d_text;          // Input text buffer
    int *d_results;        // Results buffer
    size_t trieSize;       // Size of transition table
    size_t outputSize;     // Size of output table
} GPUData;

// PFAC structure
typedef struct {
    GPUData gpu_data;
    Pattern *patterns;     // Host patterns array
    int numPatterns;
    int capacity;
} PFAC;

// Function declarations
PFAC* createPFAC(int initialCapacity);
void destroyPFAC(PFAC *pfac);
int addPattern(PFAC *pfac, const Pattern *pattern);
void buildMachine(PFAC *pfac);
MatchResult* searchPatterns(PFAC *pfac, const char *text, int *numMatches);  // Diubah agar sama dengan versi sequential

// Utility functions
void loadPatternsFromFile(PFAC *pfac, const char *filename);
void printStatistics(const PFAC *pfac);

#endif // PFAC_CUH