#include "../sequential/aho_corasick.h"
#include "../include/patterns.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Queue implementation for BFS
typedef struct QueueNode {
    TrieNode *node;
    struct QueueNode *next;
} QueueNode;

typedef struct {
    QueueNode *front, *rear;
} Queue;

// Queue functions
static Queue* createQueue(void) {
    Queue *q = (Queue*)malloc(sizeof(Queue));
    if (!q) {
        fprintf(stderr, "Failed to allocate queue\n");
        exit(1);
    }
    q->front = q->rear = NULL;
    return q;
}

static void enqueue(Queue *q, TrieNode *node) {
    QueueNode *newNode = (QueueNode*)malloc(sizeof(QueueNode));
    if (!newNode) {
        fprintf(stderr, "Failed to allocate queue node\n");
        exit(1);
    }
    newNode->node = node;
    newNode->next = NULL;

    if (!q->rear) {
        q->front = q->rear = newNode;
        return;
    }
    q->rear->next = newNode;
    q->rear = newNode;
}

static TrieNode* dequeue(Queue *q) {
    if (!q->front) return NULL;

    QueueNode *temp = q->front;
    TrieNode *node = temp->node;
    q->front = temp->next;

    if (!q->front) q->rear = NULL;
    free(temp);
    return node;
}

// Create new trie node
static TrieNode* createTrieNode(void) {
    TrieNode *node = (TrieNode*)malloc(sizeof(TrieNode));
    if (!node) {
        fprintf(stderr, "Failed to allocate trie node\n");
        exit(1);
    }

    memset(node->children, 0, sizeof(node->children));
    node->fail = NULL;
    node->patterns = NULL;
    node->numPatterns = 0;
    node->capacity = 0;
    
    return node;
}

// Initialize Aho-Corasick automaton
AhoCorasick* createAhoCorasick(int initialCapacity) {
    AhoCorasick *ac = (AhoCorasick*)malloc(sizeof(AhoCorasick));
    if (!ac) {
        fprintf(stderr, "Failed to allocate Aho-Corasick structure\n");
        exit(1);
    }

    ac->root = createTrieNode();
    ac->patterns = (Pattern*)malloc(initialCapacity * sizeof(Pattern));
    if (!ac->patterns) {
        fprintf(stderr, "Failed to allocate patterns array\n");
        free(ac);
        exit(1);
    }

    ac->numPatterns = 0;
    ac->capacity = initialCapacity;
    return ac;
}

// Add a pattern to the trie
int addPattern(AhoCorasick *ac, const Pattern *pattern) {
    if (ac->numPatterns >= ac->capacity) {
        int newCapacity = ac->capacity * 2;
        Pattern *newPatterns = (Pattern*)realloc(ac->patterns, newCapacity * sizeof(Pattern));
        if (!newPatterns) {
            fprintf(stderr, "Failed to reallocate patterns array\n");
            return 0;
        }
        ac->patterns = newPatterns;
        ac->capacity = newCapacity;
    }

    // Copy pattern to patterns array
    memcpy(&ac->patterns[ac->numPatterns], pattern, sizeof(Pattern));
    
    // Add pattern to trie
    TrieNode *current = ac->root;
    const char *pat = pattern->pattern;
    
    for (int i = 0; pat[i]; i++) {
        unsigned char ch = (unsigned char)pat[i];
        if (!current->children[ch]) {
            current->children[ch] = createTrieNode();
        }
        current = current->children[ch];
    }

    // Add pattern to the node's pattern list
    if (current->numPatterns >= current->capacity) {
        int newCapacity = current->capacity == 0 ? 1 : current->capacity * 2;
        Pattern *newPatterns = (Pattern*)realloc(current->patterns, newCapacity * sizeof(Pattern));
        if (!newPatterns) {
            fprintf(stderr, "Failed to reallocate node patterns array\n");
            return 0;
        }
        current->patterns = newPatterns;
        current->capacity = newCapacity;
    }

    memcpy(&current->patterns[current->numPatterns], pattern, sizeof(Pattern));
    current->numPatterns++;
    ac->numPatterns++;
    
    return 1;
}

// Build failure links using BFS
void buildFailureLinks(AhoCorasick *ac) {
    Queue *q = createQueue();
    TrieNode *current;
    
    // Set failure for depth 1 nodes to root
    for (int i = 0; i < MAX_CHAR; i++) {
        TrieNode *child = ac->root->children[i];
        if (child) {
            child->fail = ac->root;
            enqueue(q, child);
        }
    }

    // Process all nodes
    while ((current = dequeue(q)) != NULL) {
        for (int i = 0; i < MAX_CHAR; i++) {
            TrieNode *child = current->children[i];
            if (!child) continue;

            // Find failure node
            TrieNode *failNode = current->fail;
            while (failNode && !failNode->children[i]) {
                failNode = failNode->fail;
            }

            child->fail = failNode ? failNode->children[i] : ac->root;

            // Add patterns from failure node
            if (child->fail->numPatterns > 0) {
                int newSize = child->numPatterns + child->fail->numPatterns;
                if (newSize > child->capacity) {
                    Pattern *newPatterns = (Pattern*)realloc(child->patterns, newSize * sizeof(Pattern));
                    if (!newPatterns) {
                        fprintf(stderr, "Failed to reallocate patterns in failure link\n");
                        continue;
                    }
                    child->patterns = newPatterns;
                    child->capacity = newSize;
                }

                // Copy patterns from failure node
                memcpy(&child->patterns[child->numPatterns], 
                       child->fail->patterns, 
                       child->fail->numPatterns * sizeof(Pattern));
                child->numPatterns = newSize;
            }

            enqueue(q, child);
        }
    }

    free(q);
}

// Search for patterns in text
MatchResult* searchPatterns(const AhoCorasick *ac, const char *text, int *numMatches) {
    MatchResult *results = NULL;
    int resultCapacity = 16;
    *numMatches = 0;

    results = (MatchResult*)malloc(resultCapacity * sizeof(MatchResult));
    if (!results) {
        fprintf(stderr, "Failed to allocate results array\n");
        return NULL;
    }

    const TrieNode *current = ac->root;
    int textLen = strlen(text);

    for (int i = 0; i < textLen; i++) {
        unsigned char ch = (unsigned char)text[i];

        // Follow failure links until match is found
        while (current != ac->root && !current->children[ch]) {
            current = current->fail;
        }

        current = current->children[ch] ? current->children[ch] : ac->root;

        // Check for matches
        if (current->numPatterns > 0) {
            for (int j = 0; j < current->numPatterns; j++) {
                const Pattern *pattern = &current->patterns[j];
                int patternLen = strlen(pattern->pattern);
                int startPos = i - patternLen + 1;

                // Ensure capacity
                if (*numMatches >= resultCapacity) {
                    resultCapacity *= 2;
                    MatchResult *newResults = (MatchResult*)realloc(results, resultCapacity * sizeof(MatchResult));
                    if (!newResults) {
                        fprintf(stderr, "Failed to reallocate results array\n");
                        free(results);
                        return NULL;
                    }
                    results = newResults;
                }

                // Add match to results
                results[*numMatches].position = startPos;
                results[*numMatches].type = pattern->type;
                strncpy(results[*numMatches].pattern, pattern->pattern, MAX_PATTERN_LENGTH - 1);
                results[*numMatches].pattern[MAX_PATTERN_LENGTH - 1] = '\0';
                (*numMatches)++;
            }
        }
    }

    return results;
}

// Helper function for recursive node freeing
static void freeNode(TrieNode *node) {
    if (!node) return;
    for (int i = 0; i < MAX_CHAR; i++) {
        if (node->children[i]) {
            freeNode(node->children[i]);
        }
    }
    free(node->patterns);
    free(node);
}

// Clean up
void destroyAhoCorasick(AhoCorasick *ac) {
    if (!ac) return;
    freeNode(ac->root);
    free(ac->patterns);
    free(ac);
}

// Helper function for recursive node counting
static void countNodes(const TrieNode *node, int depth, int *totalNodes, int *maxDepth) {
    if (!node) return;
    (*totalNodes)++;
    if (depth > *maxDepth) *maxDepth = depth;
    
    for (int i = 0; i < MAX_CHAR; i++) {
        if (node->children[i]) {
            countNodes(node->children[i], depth + 1, totalNodes, maxDepth);
        }
    }
}

// Print statistics about the trie
void printTrieStatistics(const AhoCorasick *ac) {
    int totalNodes = 0;
    int maxDepth = 0;
    
    countNodes(ac->root, 0, &totalNodes, &maxDepth);
    
    printf("Trie Statistics:\n");
    printf("Total patterns: %d\n", ac->numPatterns);
    printf("Total nodes: %d\n", totalNodes);
    printf("Maximum depth: %d\n", maxDepth);
}

// Load patterns from file
void loadPatternsFromFile(AhoCorasick *ac, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Parse pattern type and pattern
        char *type = strtok(line, "|");
        char *patternStr = strtok(NULL, "|");
        char *priorityStr = strtok(NULL, "|");
        
        if (!type || !patternStr || !priorityStr) continue;
        
        Pattern pattern;
        strncpy(pattern.pattern, patternStr, MAX_PATTERN_LENGTH - 1);
        pattern.pattern[MAX_PATTERN_LENGTH - 1] = '\0';
        pattern.type = getPatternType(type);
        pattern.priority = atoi(priorityStr);
        
        addPattern(ac, &pattern);
    }

    fclose(file);
}

void printPattern(const Pattern *pattern) {
    printf("Pattern: %s (Type: %s, Priority: %d)\n",
           pattern->pattern,
           getPatternTypeString(pattern->type),
           pattern->priority);
}

void printMatchResult(const MatchResult *result) {
    printf("Match found at position %d: %s (Type: %s)\n",
           result->position,
           result->pattern,
           getPatternTypeString(result->type));
}