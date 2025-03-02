#ifndef PATTERNS_H
#define PATTERNS_H

#include "common.h"

// Function declarations for pattern type handling
PatternType getPatternType(const char* patternType);
const char* getPatternTypeString(PatternType type);

// Fungsi untuk loading patterns ke array (berbeda dengan yang di aho_corasick.h)
int loadPatternsToArray(const char* filename, Pattern* patterns, int* numPatterns);

#endif // PATTERNS_H