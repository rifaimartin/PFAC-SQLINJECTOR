#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <queue>
#include <cuda_runtime.h>
#include <stdint.h>
#include <regex>

using namespace std;

#define ALPHABET_SIZE 128
#define MAX_NODES     8192    // adjust as needed
#define MAX_PATTERNS  250     // adjust as needed
#define THREADS_PER_BLOCK 256

// Sparse transition representation
struct SparseTransition {
    unsigned char character;
    int next_state;
};

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << endl;        \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Device memory for sparse transition table - MOVED TO GLOBAL MEMORY INSTEAD OF CONSTANT
__device__ __constant__ int d_pattern_weights[MAX_PATTERNS];    // Keep this in constant memory

// Optimized PFAC kernel - one thread per query
__global__ void PFACKernel(
    const char* __restrict__ d_queries,
    const int*  __restrict__ d_offsets,
    const int*  __restrict__ d_lengths,
    int          numQueries,
    int*        __restrict__ d_risk_scores,
    // Pass tables as kernel parameters instead of using constant memory
    const int* __restrict__ d_transition_offsets,
    const SparseTransition* __restrict__ d_transitions,
    const uint64_t* __restrict__ d_match_masks)
{
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= numQueries) return;
    
    // Get query details
    int offset = d_offsets[query_idx];
    int length = d_lengths[query_idx];
    
    // Thread-local variables to store matched patterns
    // Using register memory for highest performance
    uint64_t matched_low = 0;
    uint64_t matched_high = 0;
    int risk_score = 0;
    
    // Process each starting position
    for (int start_pos = 0; start_pos < length; ++start_pos) {
        int state = 0;  // Start from root
        
        // Process characters from this position
        for (int i = start_pos; i < length; ++i) {
            unsigned char c = (unsigned char)d_queries[offset + i];
            if (c >= ALPHABET_SIZE) break;
            
            // Find transition using sparse representation
            int next_state = -1;
            int transition_start = d_transition_offsets[state];
            int transition_end = d_transition_offsets[state + 1];
            
            // Linear search through sparse transitions
            for (int t = transition_start; t < transition_end; ++t) {
                if (d_transitions[t].character == c) {
                    next_state = d_transitions[t].next_state;
                    break;
                }
            }
            
            // If no transition found, break
            if (next_state == -1) break;
            state = next_state;
            
            // Check for matches at this state
            uint64_t new_matches_low = d_match_masks[state * 2] & ~matched_low;
            uint64_t new_matches_high = d_match_masks[state * 2 + 1] & ~matched_high;
            
            // Process new low matches
            while (new_matches_low) {
                int pattern_id = __ffsll(new_matches_low) - 1;
                risk_score += d_pattern_weights[pattern_id];
                new_matches_low &= (new_matches_low - 1); // Clear lowest bit
            }
            
            // Process new high matches
            while (new_matches_high) {
                int pattern_id = __ffsll(new_matches_high) - 1 + 64;
                risk_score += d_pattern_weights[pattern_id];
                new_matches_high &= (new_matches_high - 1); // Clear lowest bit
            }
            
            // Update match masks
            matched_low |= d_match_masks[state * 2];
            matched_high |= d_match_masks[state * 2 + 1];
        }
        
        // Reset match tracking for next starting position
        matched_low = 0;
        matched_high = 0;
    }
    
    // Write final risk score for this query
    d_risk_scores[query_idx] = risk_score;
}

// CPU Trie Node for building automaton
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    vector<int> pattern_ids;
    int id;
    
    TrieNode(int id): id(id) {}
};

class PFACTrie {
public:
    PFACTrie() { 
        root = new TrieNode(0); 
        nodes.push_back(root);
    }
    
    ~PFACTrie() {
        for (auto node : nodes) {
            delete node;
        }
    }

    void insert(const string& pat, int pattern_id) {
        TrieNode* current = root;
        
        for (char ch : pat) {
            if (!current->children.count(ch)) {
                TrieNode* newNode = new TrieNode(nodes.size());
                current->children[ch] = newNode;
                nodes.push_back(newNode);
            }
            current = current->children[ch];
        }
        
        current->pattern_ids.push_back(pattern_id);
    }
    
    // Build sparse transition tables
    void buildSparseTransitionTables(
        vector<int>& transition_offsets,
        vector<SparseTransition>& transitions,
        vector<uint64_t>& match_masks)
    {
        int numNodes = nodes.size();
        transition_offsets.resize(numNodes + 1);
        match_masks.resize(numNodes * 2, 0); // 2 uint64_t per node
        
        // Count transitions per node
        int totalTransitions = 0;
        for (int i = 0; i < numNodes; i++) {
            transition_offsets[i] = totalTransitions;
            totalTransitions += nodes[i]->children.size();
        }
        transition_offsets[numNodes] = totalTransitions;
        
        // Allocate transitions
        transitions.resize(totalTransitions);
        
        // Fill transitions and match masks
        for (int i = 0; i < numNodes; i++) {
            TrieNode* node = nodes[i];
            int transIdx = transition_offsets[i];
            
            // Add transitions for this node
            for (auto& kv : node->children) {
                unsigned char ch = kv.first;
                TrieNode* child = kv.second;
                transitions[transIdx].character = ch;
                transitions[transIdx].next_state = child->id;
                transIdx++;
            }
            
            // Set match masks
            for (int pid : node->pattern_ids) {
                if (pid < 64) {
                    match_masks[i * 2] |= (1ULL << pid);
                } else {
                    match_masks[i * 2 + 1] |= (1ULL << (pid - 64));
                }
            }
        }
    }
    
    int nodeCount() const { return nodes.size(); }

private:
    TrieNode* root;
    vector<TrieNode*> nodes;
};

// Text normalization
string normalize(const string &s) {
    string r = s;
    transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

// Updated risk classification
string classifyRisk(int score) {
    if (score <= 20)   return "low";       // 0-20 (basic injection attempts)
    if (score <= 45)   return "medium";    // 21-45 (union attacks, info disclosure)  
    if (score <= 75)   return "high";      // 46-75 (destructive operations)
    return "critical";                     // 76+ (system-level attacks)
}

// Function to read patterns from file
vector<string> readPatternsFromFile(const string& filename) {
    vector<string> patterns;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open pattern file: " << filename << endl;
        return patterns;
    }
    
    // Read whole file into a string
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    // Process with regex to extract patterns more reliably
    regex patternRegex("\"([^\"]+)\"");
    auto matches_begin = sregex_iterator(content.begin(), content.end(), patternRegex);
    auto matches_end = sregex_iterator();
    
    for (sregex_iterator i = matches_begin; i != matches_end; ++i) {
        smatch match = *i;
        string pattern = match[1].str(); // Get the content inside the quotes
        
        // Remove trailing comma if present
        if (!pattern.empty() && pattern.back() == ',') {
            pattern.pop_back();
        }
        
        // Add to patterns list
        if (!pattern.empty()) {
            patterns.push_back(pattern);
        }
    }
    
    cout << "Extracted " << patterns.size() << " patterns using regex" << endl;
    
    // If no patterns found, try a different approach
    if (patterns.empty()) {
        file.open(filename);
        string line;
        while (getline(file, line)) {
            // Skip empty lines and comment lines
            if (line.empty() || line[0] == '#') continue;
            
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            // Remove quotes and comma if present
            if (line.size() >= 2 && line.front() == '"' && 
                (line.back() == '"' || (line.size() >= 3 && line[line.size()-2] == '"' && line.back() == ','))) {
                
                // Remove opening quote
                line = line.substr(1);
                
                // Remove closing quote and optional comma
                if (line.back() == ',') {
                    line = line.substr(0, line.size() - 2);  // Remove both " and ,
                } else if (line.back() == '"') {
                    line = line.substr(0, line.size() - 1);  // Remove just "
                }
            } else if (line.back() == ',') {
                // If only comma present at end with no quotes
                line = line.substr(0, line.size() - 1);
            }
            
            // Add pattern if it's not empty after processing
            if (!line.empty()) {
                patterns.push_back(line);
            }
        }
        file.close();
        cout << "Secondary extraction found " << patterns.size() << " patterns" << endl;
    }
    
    return patterns;
}

void printUsage(const char* programName) {
    cout << "Usage: " << programName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -d, --dataset <file>   CSV dataset file to process (default: sql_dataset_Critical_10000.csv)" << endl;
    cout << "  -p, --patterns <file>  Pattern file to use (default: patterns.txt)" << endl;
    cout << "  -o, --output <file>    Output file for results (default: results.txt)" << endl;
    cout << "  -h, --help             Show this help message" << endl;
}

int main(int argc, char** argv) {
    // Default filenames
    string datasetFile = "sql_dataset_Critical_10000";

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-d" || arg == "--dataset")
        {
            datasetFile = argv[++i];
        }
        else
        {
            cerr << "Unknown option: " << arg << endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    ofstream out("result_PFAC_" + datasetFile + ".txt");
    streambuf* coutbuf = cout.rdbuf(); 
    cout.rdbuf(out.rdbuf());

    // 1. Load patterns
    vector<string> rawPatterns;
    
    string patternFile = "patterns.txt";
    
    ifstream testFile(patternFile);
    if (testFile.is_open()) {
        testFile.close();
        rawPatterns = readPatternsFromFile(patternFile);
        if (!rawPatterns.empty()) {
            cout << "Successfully read " << rawPatterns.size() << " patterns from " << patternFile << endl;
        } else {
            cout << "Warning: Could not extract any patterns from " << patternFile << endl;
        }
    } else {
        cout << "Warning: Could not open pattern file " << patternFile << ". Using default patterns." << endl;
    }

    int P = rawPatterns.size();
    
    // Display first few patterns
    cout << "First " << min(10, P) << " patterns:" << endl;
    for (int i = 0; i < min(10, P); i++) {
        cout << i+1 << ". " << rawPatterns[i] << endl;
    }
    if (P > 10) {
        cout << "... and " << (P-10) << " more patterns" << endl;
    }

    vector<string> patterns(P);
    vector<int> weights(P);
    
    for (int i = 0; i < P; ++i) {
        patterns[i] = normalize(rawPatterns[i]);
        const auto &pat = patterns[i];

         // CRITICAL RISK (80-100 points) - System-level attacks
    if (pat.find("xp_cmdshell") != string::npos ||
        pat.find("exec xp_") != string::npos ||
        pat.find("into outfile") != string::npos ||
        pat.find("load_file") != string::npos ||
        pat.find("load data infile") != string::npos ||
        pat.find("/etc/passwd") != string::npos ||
        pat.find("shell.php") != string::npos ||
        pat.find("net user hack") != string::npos ||
        pat.find("lambda_async") != string::npos ||
        pat.find("create user") != string::npos ||
        pat.find("grant all privileges") != string::npos ||
        pat.find("create trigger") != string::npos) {
        weights[i] = 90;
    }
    
    // HIGH RISK (40-60 points) - Destructive operations
    else if (pat.find("drop table") != string::npos ||
             pat.find("drop database") != string::npos ||
             pat.find("drop procedure") != string::npos ||
             pat.find("delete from") != string::npos ||
             pat.find("alter table") != string::npos ||
             pat.find("insert into") != string::npos ||
             pat.find("update users") != string::npos ||
             pat.find("exec(@cmd)") != string::npos ||
             pat.find("exec sp_") != string::npos ||
             pat.find("exec master") != string::npos ||
             pat.find("pg_sleep") != string::npos ||
             pat.find("sleep(") != string::npos ||
             pat.find("waitfor delay") != string::npos ||
             pat.find("benchmark(") != string::npos ||
             pat.find("@@version") != string::npos ||
             pat.find("mysql.user") != string::npos) {
        weights[i] = 50;
    }
    
    // MEDIUM-HIGH RISK (25-35 points) - Advanced injection
    else if (pat.find("union select") != string::npos ||
             pat.find("union all select") != string::npos ||
             pat.find("information_schema") != string::npos ||
             pat.find("password from users") != string::npos ||
             pat.find("from users") != string::npos ||
             pat.find("table_name") != string::npos ||
             pat.find("table_schema") != string::npos ||
             pat.find("count(*)") != string::npos ||
             pat.find("group by") != string::npos ||
             pat.find("having") != string::npos) {
        weights[i] = 30;
    }
    
    // MEDIUM RISK (15-20 points) - Union and basic attacks  
    else if (pat.find("union") != string::npos ||
             pat.find("select from") != string::npos ||
             pat.find("' select") != string::npos ||
             pat.find("username") != string::npos ||
             pat.find("admin' --") != string::npos ||
             pat.find("convert(") != string::npos ||
             pat.find("cast(") != string::npos ||
             pat.find("md5(") != string::npos ||
             pat.find("limit 1") != string::npos) {
        weights[i] = 18;
    }
    
    // LOW-MEDIUM RISK (8-12 points) - Basic injection patterns
    else if (pat.find("1=1") != string::npos ||
             pat.find("' or") != string::npos ||
             pat.find("\" or") != string::npos ||
             pat.find("' and") != string::npos ||
             pat.find("\" and") != string::npos ||
             pat.find("'='") != string::npos ||
             pat.find("\"=\"") != string::npos ||
             pat.find("true=true") != string::npos ||
             pat.find("' ||") != string::npos ||
             pat.find("\" ||") != string::npos ||
             pat.find("'a'='a") != string::npos ||
             pat.find("'x'='x") != string::npos ||
             pat.find("'1'='1") != string::npos) {
        weights[i] = 10;
    }
    
    // LOW RISK (3-6 points) - Comments and basic operators
    else if (pat.find("--") != string::npos ||
             pat.find("/**/") != string::npos ||
             pat.find("/* test */") != string::npos ||
             pat.find("' =") != string::npos ||
             pat.find("\" =") != string::npos ||
             pat.find(" <=") != string::npos ||
             pat.find(" >=") != string::npos ||
             pat.find(" <>") != string::npos ||
             pat.find("= =") != string::npos ||
             pat.find("= <") != string::npos ||
             pat.find("= or") != string::npos ||
             pat.find("= ||") != string::npos) {
        weights[i] = 5;
    }
    
    // DEFAULT - Very basic patterns
    else {
        weights[i] = 1;
    }
    }
    
    // Build PFAC Trie
    PFACTrie trie;
    for (int i = 0; i < P; ++i) {
        trie.insert(patterns[i], i);
    }
    
    //  Build sparse transition tables
    vector<int> h_transition_offsets;
    vector<SparseTransition> h_transitions;
    vector<uint64_t> h_match_masks;
    trie.buildSparseTransitionTables(h_transition_offsets, h_transitions, h_match_masks);
    
    int N = trie.nodeCount();
    cout << "Trie node count: " << N << endl;
    cout << "Transitions count: " << h_transitions.size() << endl;
    cout << "Average transitions per node: " << (float)h_transitions.size() / N << endl;
    
    // Verify we don't exceed max sizes
    if (N > MAX_NODES) {
        cerr << "Error: Too many nodes in trie (" << N << "), max is " << MAX_NODES << endl;
        return EXIT_FAILURE;
    }
    
    if (P > MAX_PATTERNS) {
        cerr << "Error: Too many patterns (" << P << "), max is " << MAX_PATTERNS << endl;
        return EXIT_FAILURE;
    }
    
    // Allocate GPU memory for transition tables
    int* d_transition_offsets;
    SparseTransition* d_transitions;
    uint64_t* d_match_masks;
    
    CUDA_CHECK(cudaMalloc(&d_transition_offsets, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transitions, h_transitions.size() * sizeof(SparseTransition)));
    CUDA_CHECK(cudaMalloc(&d_match_masks, N * 2 * sizeof(uint64_t)));
    
    // Copy transition data to GPU
    CUDA_CHECK(cudaMemcpy(d_transition_offsets, h_transition_offsets.data(), 
                        (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transitions, h_transitions.data(), 
                        h_transitions.size() * sizeof(SparseTransition), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_match_masks, h_match_masks.data(), 
                        N * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_pattern_weights, weights.data(), 
                                P * sizeof(int)));
    
    ifstream infile(datasetFile+".csv");
    if (!infile.is_open()) {
        cerr << "Error: could not open CSV file.\n";
        return EXIT_FAILURE;
    }
    
    string line;
    vector<string> queries;
    vector<string> expected;
    vector<string> originalQueries;
    
    // Skip header
    getline(infile, line);
    
    int lineNum = 0;
    while (getline(infile, line)) {
        lineNum++;
        if (line.empty()) continue;
        
        // Robust CSV parsing untuk handle quoted strings dengan koma
        vector<string> fields;
        string current_field = "";
        bool in_quotes = false;
        
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            
            if (c == '"') {
                in_quotes = !in_quotes;
            }
            else if (c == ',' && !in_quotes) {
                // Field separator - simpan field dan reset
                fields.push_back(current_field);
                current_field = "";
            }
            else {
                current_field += c;
            }
        }
        
        fields.push_back(current_field);
        
        if (fields.size() >= 2) {
            string query_field = fields[0];
            string risk_field = fields[1];
            
            if (query_field.length() >= 2 && query_field.front() == '"' && query_field.back() == '"') {
                query_field = query_field.substr(1, query_field.length() - 2);
            }
            if (risk_field.length() >= 2 && risk_field.front() == '"' && risk_field.back() == '"') {
                risk_field = risk_field.substr(1, risk_field.length() - 2);
            }
            
            query_field.erase(0, query_field.find_first_not_of(" \t\r\n"));
            query_field.erase(query_field.find_last_not_of(" \t\r\n") + 1);
            risk_field.erase(0, risk_field.find_first_not_of(" \t\r\n"));
            risk_field.erase(risk_field.find_last_not_of(" \t\r\n") + 1);
            
            if (risk_field != "low" && risk_field != "medium" && 
                risk_field != "high" && risk_field != "critical") {
                // Jika bukan label valid, skip entry ini
                continue;
            }
            
            originalQueries.push_back(query_field);
            queries.push_back(normalize(query_field));
            expected.push_back(risk_field);
        }
    }
    infile.close();
    
    int Q = queries.size();
    cout << "Loaded " << Q << " queries." << endl;

    // Prepare query data for GPU
    vector<int> h_offsets(Q), h_lengths(Q);
    int totalLen = 0;
    for (int i = 0; i < Q; ++i) {
        h_offsets[i] = totalLen;
        h_lengths[i] = queries[i].size();
        totalLen += h_lengths[i];
    }
    
    vector<char> h_buffer(totalLen);
    for (int i = 0; i < Q; ++i) {
        memcpy(&h_buffer[h_offsets[i]], queries[i].data(), h_lengths[i]);
    }

    // Allocate GPU memory
    char* d_queries;
    int* d_offsets;
    int* d_lengths;
    int* d_risk_scores;
    
    CUDA_CHECK(cudaMalloc(&d_queries, totalLen * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_offsets, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_risk_scores, Q * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_queries, h_buffer.data(), totalLen * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_risk_scores, 0, Q * sizeof(int)));

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (Q + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "Launching kernel with " << blocks << " blocks and " 
         << threadsPerBlock << " threads per block..." << endl;
    
    cudaEventRecord(start);
    
    // Launch kernel with one thread per query
    PFACKernel<<<blocks, threadsPerBlock>>>(
        d_queries, d_offsets, d_lengths, Q, d_risk_scores,
        d_transition_offsets, d_transitions, d_match_masks);
    
    cudaEventRecord(stop);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    vector<int> h_results(Q);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_risk_scores, Q * sizeof(int), cudaMemcpyDeviceToHost));

    int correct = 0;
    int zeroScores = 0;
    
    for (int i = 0; i < Q; ++i) {
        string compRisk = classifyRisk(h_results[i]);
        bool match = (compRisk == expected[i]);
        if (match) ++correct;
        if (h_results[i] == 0) ++zeroScores;
        
        // Print first 10 
        if (i < 10) {
            cout << "Query " << i << ": \"" << originalQueries[i] << "\"" << endl;
            cout << "  Normalized: \"" << queries[i] << "\"" << endl;
            cout << "  Score: " << h_results[i] << ", computed risk: " << compRisk 
                 << ", expected: " << expected[i] << (match ? " [OK]" : " [Mismatch]") << endl;
        }
    }
    
    double accuracy = Q ? (100.0 * correct / Q) : 0.0;
    cout << "\nTotal: " << Q << ", Correct: " << correct
         << ", Accuracy: " << accuracy << "%" << endl;
    cout << "Queries with zero score: " << zeroScores << " (" 
         << (100.0 * zeroScores / Q) << "%)" << endl;
    cout << "Optimized Sparse PFAC kernel execution time: " << milliseconds << " ms" << endl;

    cudaFree(d_queries);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_risk_scores);
    cudaFree(d_transition_offsets);
    cudaFree(d_transitions);
    cudaFree(d_match_masks);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout.rdbuf(coutbuf); 
    return 0;
}