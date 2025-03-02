#include "include/patterns.h"
#include <string.h>

PatternType getPatternType(const char* patternType) {
    if (strcmp(patternType, "UNION_BASED") == 0) return UNION_BASED;
    if (strcmp(patternType, "ERROR_BASED") == 0) return ERROR_BASED;
    if (strcmp(patternType, "BOOLEAN_BASED") == 0) return BOOLEAN_BASED;
    if (strcmp(patternType, "TIME_BASED") == 0) return TIME_BASED;
    if (strcmp(patternType, "STACKED_QUERIES") == 0) return STACKED_QUERIES;
    if (strcmp(patternType, "COMMENT_BASED") == 0) return COMMENT_BASED;
    return UNKNOWN;
}

const char* getPatternTypeString(PatternType type) {
    switch(type) {
        case UNION_BASED: return "UNION_BASED";
        case ERROR_BASED: return "ERROR_BASED";
        case BOOLEAN_BASED: return "BOOLEAN_BASED";
        case TIME_BASED: return "TIME_BASED";
        case STACKED_QUERIES: return "STACKED_QUERIES";
        case COMMENT_BASED: return "COMMENT_BASED";
        default: return "UNKNOWN";
    }
}

int loadPatternsToArray(const char* filename, Pattern* patterns, int* numPatterns) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 0;
    }

    char line[MAX_LINE_LENGTH];
    *numPatterns = 0;

    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Parse line
        char *type = strtok(line, "|");
        char *patternStr = strtok(NULL, "|");
        char *priorityStr = strtok(NULL, "|");
        
        if (!type || !patternStr || !priorityStr) continue;

        strncpy(patterns[*numPatterns].pattern, patternStr, MAX_PATTERN_LENGTH - 1);
        patterns[*numPatterns].pattern[MAX_PATTERN_LENGTH - 1] = '\0';
        patterns[*numPatterns].type = getPatternType(type);
        patterns[*numPatterns].priority = atoi(priorityStr);
        
        (*numPatterns)++;
    }

    fclose(file);
    return 1;
}