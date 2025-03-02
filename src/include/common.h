#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constants
#define MAX_CHAR 256
#define MAX_PATTERN_LENGTH 100
#define MAX_PATTERNS 1000
#define MAX_LINE_LENGTH 1024

// Pattern types
typedef enum {
    UNION_BASED,
    ERROR_BASED,
    BOOLEAN_BASED,
    TIME_BASED,
    STACKED_QUERIES,
    COMMENT_BASED,
    UNKNOWN
} PatternType;

// Pattern structure
typedef struct {
    char pattern[MAX_PATTERN_LENGTH];
    PatternType type;
    int priority;
} Pattern;

// Match result structure
typedef struct {
    char pattern[MAX_PATTERN_LENGTH];
    int position;
    PatternType type;
} MatchResult;

#endif // COMMON_H