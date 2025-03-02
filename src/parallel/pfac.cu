#include "pfac.cuh"
#include "../include/patterns.h"
#include <stdio.h>
#include <time.h>

// CUDA kernel for pattern matching
__global__ void matchPatternsKernel(
    const char *text,
    size_t textLen,
    const int *trie,
    const int *output,
    size_t outputSize,
    int *results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= textLen) return;

    // Each thread processes starting from its position
    int currentState = 0;
    int pos = tid;

    while (pos < textLen) {
        // Get next character
        unsigned char ch = text[pos];
        
        // Get next state from trie
        int nextState = trie[currentState * MAX_CHAR + ch];
        
        if (nextState == 0) break;  // No transition
        
        // Check for matches
        if (output[nextState]) {
            atomicOr(&results[tid], output[nextState]);
        }
        
        currentState = nextState;
        pos++;
    }
}

PatternType getPatternType(const char* patternType) {
    if (strcmp(patternType, "UNION_BASED") == 0) return UNION_BASED;
    if (strcmp(patternType, "ERROR_BASED") == 0) return ERROR_BASED;
    if (strcmp(patternType, "BOOLEAN_BASED") == 0) return BOOLEAN_BASED;
    if (strcmp(patternType, "TIME_BASED") == 0) return TIME_BASED;
    if (strcmp(patternType, "STACKED_QUERIES") == 0) return STACKED_QUERIES;
    if (strcmp(patternType, "COMMENT_BASED") == 0) return COMMENT_BASED;
    return UNKNOWN;
}

// Initialize PFAC
PFAC* createPFAC(int initialCapacity) {
    PFAC *pfac = (PFAC*)malloc(sizeof(PFAC));
    if (!pfac) {
        fprintf(stderr, "Failed to allocate PFAC structure\n");
        return NULL;
    }

    pfac->patterns = (Pattern*)malloc(initialCapacity * sizeof(Pattern));
    if (!pfac->patterns) {
        fprintf(stderr, "Failed to allocate patterns array\n");
        free(pfac);
        return NULL;
    }

    pfac->numPatterns = 0;
    pfac->capacity = initialCapacity;

    // Initialize GPU data
    pfac->gpu_data.d_trie = NULL;
    pfac->gpu_data.d_output = NULL;
    pfac->gpu_data.d_text = NULL;
    pfac->gpu_data.d_results = NULL;
    pfac->gpu_data.trieSize = 0;
    pfac->gpu_data.outputSize = 0;

    return pfac;
}

// Add pattern to PFAC
int addPattern(PFAC *pfac, const Pattern *pattern) {
    if (pfac->numPatterns >= pfac->capacity) {
        int newCapacity = pfac->capacity * 2;
        Pattern *newPatterns = (Pattern*)realloc(pfac->patterns, 
                                               newCapacity * sizeof(Pattern));
        if (!newPatterns) {
            fprintf(stderr, "Failed to reallocate patterns array\n");
            return 0;
        }
        pfac->patterns = newPatterns;
        pfac->capacity = newCapacity;
    }

    memcpy(&pfac->patterns[pfac->numPatterns], pattern, sizeof(Pattern));
    pfac->numPatterns++;
    return 1;
}

// Build trie and output tables for GPU
void buildMachine(PFAC *pfac) {
    // Calculate required sizes
    size_t maxStates = 1;  // Start with root state
    for (int i = 0; i < pfac->numPatterns; i++) {
        maxStates += strlen(pfac->patterns[i].pattern);
    }

    // Allocate host memory for tables
    int *trie = (int*)calloc(maxStates * MAX_CHAR, sizeof(int));
    int *output = (int*)calloc(maxStates, sizeof(int));

    // Build trie and output tables
    size_t currentState = 1;  // State 0 is root
    for (int i = 0; i < pfac->numPatterns; i++) {
        const char *pattern = pfac->patterns[i].pattern;
        int state = 0;

        for (int j = 0; pattern[j]; j++) {
            unsigned char ch = pattern[j];
            if (!trie[state * MAX_CHAR + ch]) {
                trie[state * MAX_CHAR + ch] = currentState++;
            }
            state = trie[state * MAX_CHAR + ch];
        }

        output[state] |= (1 << i);  // Mark pattern end
    }

    // Allocate and copy to GPU
    pfac->gpu_data.trieSize = maxStates * MAX_CHAR;
    pfac->gpu_data.outputSize = maxStates;

    CHECK_CUDA_ERROR(cudaMalloc(&pfac->gpu_data.d_trie, 
                               pfac->gpu_data.trieSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&pfac->gpu_data.d_output, 
                               pfac->gpu_data.outputSize * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(pfac->gpu_data.d_trie, trie,
                               pfac->gpu_data.trieSize * sizeof(int),
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(pfac->gpu_data.d_output, output,
                               pfac->gpu_data.outputSize * sizeof(int),
                               cudaMemcpyHostToDevice));

    free(trie);
    free(output);
}

// Search for patterns in text
MatchResult* searchPatterns(PFAC *pfac, const char *text, int *numMatches) {
    size_t textLen = strlen(text);
    
    // Allocate GPU memory for text and results
    CHECK_CUDA_ERROR(cudaMalloc(&pfac->gpu_data.d_text, textLen));
    CHECK_CUDA_ERROR(cudaMalloc(&pfac->gpu_data.d_results, textLen * sizeof(int)));

    // Copy text to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(pfac->gpu_data.d_text, text, textLen,
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(pfac->gpu_data.d_results, 0, 
                               textLen * sizeof(int)));

    // Launch kernel
    int numBlocks = (textLen + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > MAX_GRID_SIZE) numBlocks = MAX_GRID_SIZE;

    matchPatternsKernel<<<numBlocks, BLOCK_SIZE>>>(
        pfac->gpu_data.d_text,
        textLen,
        pfac->gpu_data.d_trie,
        pfac->gpu_data.d_output,
        pfac->gpu_data.outputSize,
        pfac->gpu_data.d_results
    );

    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Get results
    int *h_results = (int*)malloc(textLen * sizeof(int));
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, pfac->gpu_data.d_results,
                               textLen * sizeof(int), cudaMemcpyDeviceToHost));

    // Process results
    *numMatches = 0;
    MatchResult *results = NULL;
    int resultCapacity = 16;
    results = (MatchResult*)malloc(resultCapacity * sizeof(MatchResult));

    for (size_t i = 0; i < textLen; i++) {
        if (h_results[i]) {
            for (int j = 0; j < pfac->numPatterns; j++) {
                if (h_results[i] & (1 << j)) {
                    if (*numMatches >= resultCapacity) {
                        resultCapacity *= 2;
                        MatchResult *newResults = (MatchResult*)realloc(
                            results, resultCapacity * sizeof(MatchResult));
                        if (!newResults) {
                            fprintf(stderr, "Failed to reallocate results\n");
                            free(results);
                            free(h_results);
                            return NULL;
                        }
                        results = newResults;
                    }

                    results[*numMatches].position = i;
                    results[*numMatches].type = pfac->patterns[j].type;
                    strncpy(results[*numMatches].pattern,
                           pfac->patterns[j].pattern,
                           MAX_PATTERN_LENGTH - 1);
                    results[*numMatches].pattern[MAX_PATTERN_LENGTH - 1] = '\0';
                    (*numMatches)++;
                }
            }
        }
    }

    free(h_results);

    // Cleanup GPU memory
    CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_text));
    CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_results));
    pfac->gpu_data.d_text = NULL;
    pfac->gpu_data.d_results = NULL;

    return results;
}
// Load patterns from file
void loadPatternsFromFile(PFAC *pfac, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        
        line[strcspn(line, "\n")] = 0;
        
        char *type = strtok(line, "|");
        char *patternStr = strtok(NULL, "|");
        char *priorityStr = strtok(NULL, "|");
        
        if (!type || !patternStr || !priorityStr) continue;
        
        Pattern pattern;
        strncpy(pattern.pattern, patternStr, MAX_PATTERN_LENGTH - 1);
        pattern.pattern[MAX_PATTERN_LENGTH - 1] = '\0';
        pattern.type = getPatternType(type);
        pattern.priority = atoi(priorityStr);
        
        addPattern(pfac, &pattern);
    }

    fclose(file);
}

// Print statistics
void printStatistics(const PFAC *pfac) {
    printf("PFAC Statistics:\n");
    printf("Total patterns: %d\n", pfac->numPatterns);
    printf("Trie size: %zu\n", pfac->gpu_data.trieSize);
    printf("Output table size: %zu\n", pfac->gpu_data.outputSize);
}

// Cleanup
void destroyPFAC(PFAC *pfac) {
    if (!pfac) return;

    free(pfac->patterns);
    
    if (pfac->gpu_data.d_trie)
        CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_trie));
    if (pfac->gpu_data.d_output)
        CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_output));
    if (pfac->gpu_data.d_text)
        CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_text));
    if (pfac->gpu_data.d_results)
        CHECK_CUDA_ERROR(cudaFree(pfac->gpu_data.d_results));

    free(pfac);
}

char** readQueriesFromFile(const char* filename, int* numQueries) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    char** queries = (char**)malloc(1000 * sizeof(char*));  // Assuming max 1000 queries
    *numQueries = 0;
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Copy line
        queries[*numQueries] = (char*)malloc(strlen(line) + 1);
        strcpy(queries[*numQueries], line);
        (*numQueries)++;
    }

    fclose(file);
    return queries;
}

// Helper function untuk cleanup
void freeQueries(char** queries, int numQueries) {
    for (int i = 0; i < numQueries; i++) {
        free(queries[i]);
    }
    free(queries);
}

int main() {
    printf("Starting PFAC SQL Injection Detection Test\n");
    clock_t start, end;
    double cpu_time_used;

    // Initialize PFAC
    PFAC *pfac = createPFAC(100);
    if (!pfac) {
        fprintf(stderr, "Failed to create PFAC\n");
        return 1;
    }

    // Load patterns from file
    printf("Loading patterns from file...\n");
    loadPatternsFromFile(pfac, "../../data/sql_patterns.txt");

    // Build the pattern matching machine
    printf("\nBuilding pattern matching machine...\n");
    start = clock();
    buildMachine(pfac);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Build time: %.2f ms\n", cpu_time_used);

    // Print statistics
    printStatistics(pfac);

    // Test benign queries
    printf("\nTesting benign queries...\n");
    printf("----------------------------------------\n");
    int numBenign;
    char** benignQueries = readQueriesFromFile("../../data/test_cases/benign_queries.txt", &numBenign);
    
    if (benignQueries) {
        double totalTime = 0;
        int totalMatches = 0;

        for (int i = 0; i < numBenign; i++) {
            printf("\nQuery %d: %s\n", i+1, benignQueries[i]);
            
            int numMatches;
            start = clock();
            MatchResult *matches = searchPatterns(pfac, benignQueries[i], &numMatches);
            end = clock();
            cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
            
            totalTime += cpu_time_used;
            totalMatches += numMatches;

            if (numMatches > 0) {
                printf("Found %d SQL injection patterns:\n", numMatches);
                for (int j = 0; j < numMatches; j++) {
                    printf("- Pattern '%s' at position %d (Type: %d)\n",
                           matches[j].pattern,
                           matches[j].position,
                           matches[j].type);
                }
                free(matches);
            } else {
                printf("No SQL injection patterns detected\n");
            }
            printf("Detection time: %.2f ms\n", cpu_time_used);
        }

        printf("\nBenign Queries Summary:\n");
        printf("Total queries: %d\n", numBenign);
        printf("Average detection time: %.2f ms\n", totalTime / numBenign);
        printf("Total matches found: %d\n", totalMatches);
        
        freeQueries(benignQueries, numBenign);
    }

    // Test malicious queries
    printf("\nTesting malicious queries...\n");
    printf("----------------------------------------\n");
    int numMalicious;
    char** maliciousQueries = readQueriesFromFile("../../data/test_cases/malicious_queries.txt", &numMalicious);
    
    if (maliciousQueries) {
        double totalTime = 0;
        int totalMatches = 0;

        for (int i = 0; i < numMalicious; i++) {
            printf("\nQuery %d: %s\n", i+1, maliciousQueries[i]);
            
            int numMatches;
            start = clock();
            MatchResult *matches = searchPatterns(pfac, maliciousQueries[i], &numMatches);
            end = clock();
            cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
            
            totalTime += cpu_time_used;
            totalMatches += numMatches;

            if (numMatches > 0) {
                printf("Found %d SQL injection patterns:\n", numMatches);
                for (int j = 0; j < numMatches; j++) {
                    printf("- Pattern '%s' at position %d (Type: %d)\n",
                           matches[j].pattern,
                           matches[j].position,
                           matches[j].type);
                }
                free(matches);
            } else {
                printf("No SQL injection patterns detected\n");
            }
            printf("Detection time: %.2f ms\n", cpu_time_used);
        }

        printf("\nMalicious Queries Summary:\n");
        printf("Total queries: %d\n", numMalicious);
        printf("Average detection time: %.2f ms\n", totalTime / numMalicious);
        printf("Total matches found: %d\n", totalMatches);
        
        freeQueries(maliciousQueries, numMalicious);
    }

    // Cleanup
    destroyPFAC(pfac);
    printf("\nTest completed successfully\n");

    return 0;
}