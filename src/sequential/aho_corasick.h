#ifndef AHO_CORASICK_H
#define AHO_CORASICK_H

#include "../include/common.h"

// Forward declaration
struct TrieNode;

// Trie node structure
typedef struct TrieNode {
    struct TrieNode *children[MAX_CHAR];  // Array of child nodes
    struct TrieNode *fail;                // Failure link
    Pattern *patterns;                    // Array of matching patterns
    int numPatterns;                      // Number of patterns ending at this node
    int capacity;                         // Capacity of patterns array
} TrieNode;

// Aho-Corasick automaton structure
typedef struct {
    TrieNode *root;            // Root of the trie
    Pattern *patterns;         // Array of all patterns
    int numPatterns;          // Total number of patterns
    int capacity;             // Capacity of patterns array
} AhoCorasick;

// Function declarations - make sure these match the functions called in main.c
AhoCorasick* createAhoCorasick(int initialCapacity);
void destroyAhoCorasick(AhoCorasick *ac);
void loadPatternsFromFile(AhoCorasick *ac, const char *filename);
void buildFailureLinks(AhoCorasick *ac);
MatchResult* searchPatterns(const AhoCorasick *ac, const char *text, int *numMatches);
void printTrieStatistics(const AhoCorasick *ac);
void printMatchResult(const MatchResult *result);

#endif // AHO_CORASICK_H