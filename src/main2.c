#include "sequential/aho_corasick.h"
#include "parallel/pfac.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Struktur untuk statistik pengujian
typedef struct {
    int total_queries;
    int total_matches;
    double pattern_load_time;
    double build_time;
    double total_search_time;
    double min_search_time;
    double max_search_time;
    double avg_search_time;
} TestStats;

// Function untuk mendapatkan path
char* getExePath(void) {
    char* exe_path = "./";
    return strdup(exe_path);
}

// Function untuk menginisialisasi statistik
void initTestStats(TestStats *stats) {
    stats->total_queries = 0;
    stats->total_matches = 0;
    stats->pattern_load_time = 0;
    stats->build_time = 0;
    stats->total_search_time = 0;
    stats->min_search_time = 999999.0;
    stats->max_search_time = 0;
    stats->avg_search_time = 0;
}

// Function untuk menguji file dengan Aho-Corasick
void testWithAhoCorasick(AhoCorasick *ac, const char *filename, TestStats *stats) {
    char* exe_path = getExePath();
    char* full_path = (char*)malloc(strlen(exe_path) + strlen(filename) + 1);
    sprintf(full_path, "%s%s", exe_path, filename);

    FILE *file = fopen(full_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open test file: %s\n", full_path);
        free(full_path);
        free(exe_path);
        return;
    }

    printf("\nTesting with Aho-Corasick: %s\n", filename);
    printf("----------------------------------------\n");

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        line[strcspn(line, "\n")] = 0;

        printf("\nQuery: %s\n", line);
        
        stats->total_queries++;
        
        int numMatches;
        clock_t start = clock();
        MatchResult *matches = searchPatterns(ac, line, &numMatches);
        clock_t end = clock();
        
        double search_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        
        stats->total_matches += numMatches;
        stats->total_search_time += search_time;
        if (search_time < stats->min_search_time) stats->min_search_time = search_time;
        if (search_time > stats->max_search_time) stats->max_search_time = search_time;
        
        if (numMatches > 0) {
            printf("Found %d matches:\n", numMatches);
            for (int i = 0; i < numMatches; i++) {
                printMatchResult(&matches[i]);
            }
            free(matches);
        } else {
            printf("No matches found.\n");
        }
        
        printf("Search time: %.3f ms\n", search_time);
    }
    
    stats->avg_search_time = stats->total_search_time / stats->total_queries;
    
    fclose(file);
    free(full_path);
    free(exe_path);
}

// Function untuk menguji file dengan PFAC
void testWithPFAC(PFAC *pfac, const char *filename, TestStats *stats) {
    char* exe_path = getExePath();
    char* full_path = (char*)malloc(strlen(exe_path) + strlen(filename) + 1);
    sprintf(full_path, "%s%s", exe_path, filename);

    FILE *file = fopen(full_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open test file: %s\n", full_path);
        free(full_path);
        free(exe_path);
        return;
    }

    printf("\nTesting with PFAC: %s\n", filename);
    printf("----------------------------------------\n");

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        line[strcspn(line, "\n")] = 0;

        printf("\nQuery: %s\n", line);
        
        stats->total_queries++;
        
        int numMatches;
        MatchResult *matches;
        
        clock_t start = clock();
        searchPatterns(pfac, line, strlen(line), &numMatches, &matches);
        clock_t end = clock();
        
        double search_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        
        stats->total_matches += numMatches;
        stats->total_search_time += search_time;
        if (search_time < stats->min_search_time) stats->min_search_time = search_time;
        if (search_time > stats->max_search_time) stats->max_search_time = search_time;
        
        if (numMatches > 0) {
            printf("Found %d matches:\n", numMatches);
            for (int i = 0; i < numMatches; i++) {
                printMatchResult(&matches[i]);
            }
            free(matches);
        } else {
            printf("No matches found.\n");
        }
        
        printf("Search time: %.3f ms\n", search_time);
    }
    
    stats->avg_search_time = stats->total_search_time / stats->total_queries;
    
    fclose(file);
    free(full_path);
    free(exe_path);
}

// Function untuk menyimpan hasil perbandingan
void saveComparisonResults(const TestStats *ac_stats, const TestStats *pfac_stats, 
                         const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) return;

    fprintf(file, "Metric,Aho-Corasick,PFAC\n");
    fprintf(file, "Total Queries,%d,%d\n", ac_stats->total_queries, pfac_stats->total_queries);
    fprintf(file, "Total Matches,%d,%d\n", ac_stats->total_matches, pfac_stats->total_matches);
    fprintf(file, "Pattern Load Time (ms),%.3f,%.3f\n", ac_stats->pattern_load_time, pfac_stats->pattern_load_time);
    fprintf(file, "Build Time (ms),%.3f,%.3f\n", ac_stats->build_time, pfac_stats->build_time);
    fprintf(file, "Average Search Time (ms),%.3f,%.3f\n", ac_stats->avg_search_time, pfac_stats->avg_search_time);
    fprintf(file, "Min Search Time (ms),%.3f,%.3f\n", ac_stats->min_search_time, pfac_stats->min_search_time);
    fprintf(file, "Max Search Time (ms),%.3f,%.3f\n", ac_stats->max_search_time, pfac_stats->max_search_time);
    
    fclose(file);
}

int main(int argc, char *argv[]) {
    char* exe_path = getExePath();
    char* patterns_file = (char*)malloc(strlen(exe_path) + strlen("data/sql_patterns.txt") + 1);
    sprintf(patterns_file, "%sdata/sql_patterns.txt", exe_path);

    TestStats ac_stats, pfac_stats;
    initTestStats(&ac_stats);
    initTestStats(&pfac_stats);

    // Test Aho-Corasick
    printf("\n=== Testing Sequential Aho-Corasick Implementation ===\n");
    clock_t start = clock();
    AhoCorasick *ac = createAhoCorasick(100);
    loadPatternsFromFile(ac, patterns_file);
    clock_t end = clock();
    ac_stats.pattern_load_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    start = clock();
    buildFailureLinks(ac);
    end = clock();
    ac_stats.build_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    testWithAhoCorasick(ac, "data/test_cases/benign_queries.txt", &ac_stats);
    testWithAhoCorasick(ac, "data/test_cases/malicious_queries.txt", &ac_stats);

    // Test PFAC
    printf("\n=== Testing Parallel Failureless Aho-Corasick Implementation ===\n");
    start = clock();
    PFAC *pfac = createPFAC(100);
    loadPatternsFromFile(pfac, patterns_file);
    end = clock();
    pfac_stats.pattern_load_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    start = clock();
    buildMachine(pfac);
    end = clock();
    pfac_stats.build_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    testWithPFAC(pfac, "data/test_cases/benign_queries.txt", &pfac_stats);
    testWithPFAC(pfac, "data/test_cases/malicious_queries.txt", &pfac_stats);

    // Print and save comparison results
    printf("\n=== Performance Comparison ===\n");
    printf("Aho-Corasick:\n");
    printf("  Pattern Load Time: %.3f ms\n", ac_stats.pattern_load_time);
    printf("  Build Time: %.3f ms\n", ac_stats.build_time);
    printf("  Average Search Time: %.3f ms\n", ac_stats.avg_search_time);
    
    printf("\nPFAC:\n");
    printf("  Pattern Load Time: %.3f ms\n", pfac_stats.pattern_load_time);
    printf("  Build Time: %.3f ms\n", pfac_stats.build_time);
    printf("  Average Search Time: %.3f ms\n", pfac_stats.avg_search_time);

    saveComparisonResults(&ac_stats, &pfac_stats, "comparison_results.csv");

    // Cleanup
    destroyAhoCorasick(ac);
    destroyPFAC(pfac);
    free(patterns_file);
    free(exe_path);

    return 0;
}