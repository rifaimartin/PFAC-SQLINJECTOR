#include "sequential/aho_corasick.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

// Struktur untuk detail profiling per pattern type
typedef struct {
    int matches;
    double total_time;
    int pattern_count;
} PatternTypeStats;

// Struktur untuk menyimpan semua statistik
typedef struct {
    // Basic stats
    int total_queries;
    int total_matches;
    double total_time;
    double min_time;
    double max_time;
    double avg_time;
    
    // Pattern type specific stats
    PatternTypeStats type_stats[6];  // Satu untuk setiap PatternType
    
    // Memory stats
    size_t memory_used;
    
    // Build stats
    double pattern_load_time;
    double trie_build_time;
} ProfilingStats;

// Get high precision time in milliseconds using Windows API
double getCurrentTimeMs() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return (double)start.QuadPart * 1000.0 / (double)frequency.QuadPart;
}

// Initialize profiling stats
void initProfilingStats(ProfilingStats *stats) {
    memset(stats, 0, sizeof(ProfilingStats));
    stats->min_time = 999999.999;  // High initial value for min
}

// Print profiling results in detail
void printProfilingResults(const ProfilingStats *stats, const char *test_type) {
    printf("\n=== Profiling Results for %s ===\n", test_type);
    
    printf("\nBasic Statistics:\n");
    printf("Total Queries: %d\n", stats->total_queries);
    printf("Total Matches: %d\n", stats->total_matches);
    printf("Average Matches per Query: %.2f\n", 
           stats->total_queries > 0 ? (double)stats->total_matches/stats->total_queries : 0.0);

    printf("\nTiming Statistics:\n");
    printf("Total Processing Time: %.3f ms\n", stats->total_time);
    printf("Average Time per Query: %.3f ms\n", 
           stats->total_queries > 0 ? stats->total_time/stats->total_queries : 0.0);
    printf("Min Processing Time: %.3f ms\n", stats->min_time);
    printf("Max Processing Time: %.3f ms\n", stats->max_time);
    
    printf("\nPattern Type Statistics:\n");
    const char* type_names[] = {
        "UNION_BASED", "ERROR_BASED", "BOOLEAN_BASED",
        "TIME_BASED", "STACKED_QUERIES", "COMMENT_BASED"
    };
    for (int i = 0; i < 6; i++) {
        if (stats->type_stats[i].matches > 0) {
            double avg_time = stats->type_stats[i].pattern_count > 0 ? 
                            stats->type_stats[i].total_time / stats->type_stats[i].pattern_count : 0;
            printf("- %s:\n", type_names[i]);
            printf("  * Matches: %d\n", stats->type_stats[i].matches);
            printf("  * Average time: %.3f ms\n", avg_time);
        }
    }

    printf("\nBuild Statistics:\n");
    printf("Pattern Load Time: %.3f ms\n", stats->pattern_load_time);
    printf("Trie Build Time: %.3f ms\n", stats->trie_build_time);
    printf("Memory Used: %zu bytes\n", stats->memory_used);
}

void processQuery(AhoCorasick *ac, const char *query, ProfilingStats *stats) {
    if (!query || query[0] == '#' || query[0] == '\n') return;

    stats->total_queries++;
    
    int numMatches;
    double start_time = getCurrentTimeMs();
    MatchResult *matches = searchPatterns(ac, query, &numMatches);
    double end_time = getCurrentTimeMs();
    double query_time = end_time - start_time;

    // Update timing stats
    stats->total_time += query_time;
    stats->min_time = query_time < stats->min_time ? query_time : stats->min_time;
    stats->max_time = query_time > stats->max_time ? query_time : stats->max_time;
    stats->total_matches += numMatches;

    // Process matches
    if (numMatches > 0) {
        for (int i = 0; i < numMatches; i++) {
            PatternType type = matches[i].type;
            if (type >= 0 && type < 6) {
                stats->type_stats[type].matches++;
                stats->type_stats[type].total_time += query_time;
                stats->type_stats[type].pattern_count++;
            }
        }
        free(matches);
    }
}

void testWithFile(AhoCorasick *ac, const char *filename, ProfilingStats *stats) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    int lineCount = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0;  // Remove newline
        lineCount++;
        printf("\rProcessing query %d", lineCount);
        fflush(stdout);
        processQuery(ac, line, stats);
    }
    printf("\n");

    fclose(file);
}

int main() {
    ProfilingStats benign_stats, malicious_stats;
    initProfilingStats(&benign_stats);
    initProfilingStats(&malicious_stats);

    printf("Starting SQL Injection Detection Test (Sequential Version)\n");

    // Initialize and build AC
    double start_time = getCurrentTimeMs();
    AhoCorasick *ac = createAhoCorasick(100);
    
    // Load patterns
    printf("\nLoading patterns...\n");
    benign_stats.pattern_load_time = getCurrentTimeMs();
    loadPatternsFromFile(ac, "data/sql_patterns.txt");
    benign_stats.pattern_load_time = getCurrentTimeMs() - benign_stats.pattern_load_time;

    // Build trie
    printf("Building pattern matcher...\n");
    benign_stats.trie_build_time = getCurrentTimeMs();
    buildFailureLinks(ac);
    benign_stats.trie_build_time = getCurrentTimeMs() - benign_stats.trie_build_time;

    double init_time = getCurrentTimeMs() - start_time;
    printf("Initialization completed in %.3f ms\n\n", init_time);

    // Test benign queries
    printf("Testing benign queries...\n");
    testWithFile(ac, "data/test_cases/benign_queries.txt", &benign_stats);

    // Test malicious queries
    printf("\nTesting malicious queries...\n");
    testWithFile(ac, "data/test_cases/malicious_queries.txt", &malicious_stats);

    // Print results
    printProfilingResults(&benign_stats, "Benign Queries");
    printProfilingResults(&malicious_stats, "Malicious Queries");

    // Save to CSV for comparison
    FILE *csv = fopen("sequence_results.csv", "w");
    if (csv) {
        fprintf(csv, "Metric,Benign,Malicious\n");
        fprintf(csv, "Total Queries,%d,%d\n", benign_stats.total_queries, malicious_stats.total_queries);
        fprintf(csv, "Total Matches,%d,%d\n", benign_stats.total_matches, malicious_stats.total_matches);
        fprintf(csv, "Average Time (ms),%.3f,%.3f\n", 
                benign_stats.total_queries > 0 ? benign_stats.total_time/benign_stats.total_queries : 0,
                malicious_stats.total_queries > 0 ? malicious_stats.total_time/malicious_stats.total_queries : 0);
        fprintf(csv, "Pattern Load Time (ms),%.3f\n", benign_stats.pattern_load_time);
        fprintf(csv, "Trie Build Time (ms),%.3f\n", benign_stats.trie_build_time);
        fclose(csv);
    }

    // Cleanup
    destroyAhoCorasick(ac);

    return 0;
}