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
#define MAX_PATTERNS  1024     // adjust as needed
#define THREADS_PER_BLOCK 256
#define MAX_RESULTS 100000

// Macro for CUDA error checking
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << endl; exit(EXIT_FAILURE); } }

// Simplified device automaton structures - more like DNA impl
__device__ int d_transitions[MAX_NODES * ALPHABET_SIZE];  // Transition table
__device__ int d_is_match[MAX_NODES];                    // Is this state a match state?
__device__ int d_match_pattern[MAX_NODES];               // Pattern ID for match states
__device__ int d_pattern_weights[MAX_PATTERNS];          // Weight for each pattern

// Redesigned kernel: one thread per character position in query
__global__ void pfacSearchKernel(
    const char* __restrict__ d_queries,
    const int*  __restrict__ d_offsets,
    const int*  __restrict__ d_lengths,
    int          numQueries,
    int*        __restrict__ d_risk_scores,
    int*        __restrict__ d_pattern_hits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple positions in parallel across all queries
    for (int query_idx = 0; query_idx < numQueries; query_idx++) {
        int offset = d_offsets[query_idx];
        int length = d_lengths[query_idx];
        
        // Each thread takes a different starting position within the query
        for (int start_pos = tid; start_pos < length; start_pos += stride) {
            int state = 0;  // Always start from root
            
            // Process characters starting from start_pos
            for (int i = start_pos; i < length; i++) {
                unsigned char c = (unsigned char)d_queries[offset + i];
                if (c >= ALPHABET_SIZE) {
                    break;  // Invalid character
                }
                
                // Transition to next state
                state = d_transitions[state * ALPHABET_SIZE + c];
                
                // If we've reached a terminal state (no transition), stop
                if (state == -1) break;
                
                // Check if we found a match at this state
                if (d_is_match[state]) {
                    int pattern_id = d_match_pattern[state];
                    if (pattern_id >= 0 && pattern_id < MAX_PATTERNS) {
                        // Add pattern weight to risk score
                        atomicAdd(&d_risk_scores[query_idx], d_pattern_weights[pattern_id]);
                        // Record pattern hit
                        atomicAdd(&d_pattern_hits[query_idx * MAX_PATTERNS + pattern_id], 1);
                    }
                }
            }
        }
    }
}

// CPU Trie Node implementation
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    vector<int> pattern_ids;  // Pattern IDs that match at this node
    int id;                   // Node ID
    
    TrieNode(int id): id(id) {}
};

// PFAC Trie - simplified to match DNA implementation
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
        
        // Mark this node as a match for this pattern
        current->pattern_ids.push_back(pattern_id);
    }
    
    // Build transition table more like DNA implementation
    void buildTransitionTable(vector<int>& transitions, 
                             vector<int>& is_match, 
                             vector<int>& match_pattern) {
        int numNodes = nodes.size();
        
        // Initialize tables
        transitions.assign(numNodes * ALPHABET_SIZE, -1);  // -1 means no transition
        is_match.assign(numNodes, 0);
        match_pattern.assign(numNodes, -1);
        
        // Fill the transition table and match vectors
        for (int i = 0; i < numNodes; i++) {
            TrieNode* node = nodes[i];
            
            // Set transitions for this node
            for (auto& kv : node->children) {
                char ch = kv.first;
                TrieNode* child = kv.second;
                transitions[i * ALPHABET_SIZE + (unsigned char)ch] = child->id;
            }
            
            // Set match state
            if (!node->pattern_ids.empty()) {
                is_match[i] = 1;
                match_pattern[i] = node->pattern_ids[0];  // Use first pattern ID for simplicity
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

// Pattern reading function
vector<string> readPatternsFromFile(const string& filename) {
    vector<string> patterns;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open pattern file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    regex patternRegex("\"([^\"]+)\"");
    auto matches_begin = sregex_iterator(content.begin(), content.end(), patternRegex);
    auto matches_end = sregex_iterator();
    
    for (sregex_iterator i = matches_begin; i != matches_end; ++i) {
        smatch match = *i;
        string pattern = match[1].str();
        
        if (!pattern.empty() && pattern.back() == ',') {
            pattern.pop_back();
        }
        
        if (!pattern.empty()) {
            patterns.push_back(pattern);
        }
    }
    
    return patterns;
}

// Risk classification
string classifyRisk(int score) {
    if (score <= 30)   return "low";
    if (score <= 70)   return "medium";
    if (score <= 90)   return "high";
    return "critical";
}

int main() {
    ofstream out("results_optimized_PFAC.txt");
    streambuf* coutbuf = cout.rdbuf();
    cout.rdbuf(out.rdbuf());

    cout << "=== Optimized SQL Injection Detection with PFAC ===" << endl;

    // 1. Read patterns
    vector<string> rawPatterns = readPatternsFromFile("patterns.txt");
    int P = rawPatterns.size();
    cout << "Using " << P << " patterns" << endl;

    // 2. Normalize patterns and assign weights
    vector<string> patterns(P);
    vector<int> weights(P);
    
    for (int i = 0; i < P; ++i) {
        patterns[i] = normalize(rawPatterns[i]);
        const auto &pat = patterns[i];
        
        // Simplified weight assignment logic
        if (pat.find("drop table") != string::npos || pat.find("; drop") != string::npos)
            weights[i] = 100;
        else if (pat.find("; delete") != string::npos || pat.find("; insert") != string::npos)
            weights[i] = 15;
        else
            weights[i] = 10;
    }
    
    // 3. Build PFAC Trie on host
    PFACTrie trie;
    for (int i = 0; i < P; ++i) {
        trie.insert(patterns[i], i);
    }
    
    // 4. Build transition tables
    vector<int> h_transitions;
    vector<int> h_is_match;
    vector<int> h_match_pattern;
    trie.buildTransitionTable(h_transitions, h_is_match, h_match_pattern);
    
    // Copy to device using the new, simpler approach
    int N = trie.nodeCount();
    cout << "Trie node count: " << N << endl;
    CUDA_CHECK(cudaMemcpyToSymbol(d_transitions, h_transitions.data(), N * ALPHABET_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_is_match, h_is_match.data(), N * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_match_pattern, h_match_pattern.data(), N * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pattern_weights, weights.data(), P * sizeof(int)));

    // 5. Read queries
    ifstream infile("sqli_dataset_High_New_50000.csv");
    if (!infile.is_open()) {
        cerr << "Error: could not open CSV file.\n";
        return EXIT_FAILURE;
    }
    
    string line;
    vector<string> queries;
    vector<string> expected;
    
    // Skip header
    getline(infile, line);
    
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string q, expRisk, expScore;
        getline(ss, q, ',');
        getline(ss, expRisk, ',');
        getline(ss, expScore, ',');
        
        queries.push_back(normalize(q));
        expected.push_back(expRisk);
    }
    infile.close();
    
    int Q = queries.size();
    cout << "Loaded " << Q << " queries." << endl;

    // Build flat buffer for GPU
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

    // 6. Allocate device memory
    char* d_queries;
    int* d_offsets;
    int* d_lengths;
    int* d_risk_scores;
    int* d_pattern_hits;
    
    CUDA_CHECK(cudaMalloc(&d_queries, totalLen * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_offsets, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_risk_scores, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_hits, Q * MAX_PATTERNS * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_queries, h_buffer.data(), totalLen * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_risk_scores, 0, Q * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pattern_hits, 0, Q * MAX_PATTERNS * sizeof(int)));

    // 7. Launch kernel with optimized grid/block configuration
    int threadsPerBlock = THREADS_PER_BLOCK;
    
    // Calculate total number of character positions across all queries
    int totalPositions = totalLen;
    
    // Calculate grid size based on total positions
    int blocks = (totalPositions + threadsPerBlock - 1) / threadsPerBlock;
    // Limit to a reasonable number to avoid excessive blocks
    blocks = min(1024, blocks);
    
    cout << "Launching kernel with " << blocks << " blocks and " 
         << threadsPerBlock << " threads per block..." << endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel with optimized configuration
    pfacSearchKernel<<<blocks, threadsPerBlock>>>(
        d_queries, d_offsets, d_lengths, Q, d_risk_scores, d_pattern_hits);
    
    // Record stop event
    cudaEventRecord(stop);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Kernel execution time: " << milliseconds << " ms" << endl;

    // 8. Copy results back
    vector<int> h_results(Q);
    vector<int> h_pattern_hits(Q * MAX_PATTERNS);
    
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_risk_scores, Q * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pattern_hits.data(), d_pattern_hits, Q * MAX_PATTERNS * sizeof(int), cudaMemcpyDeviceToHost));

    // 9. Analyze results
    int correct = 0;
    for (int i = 0; i < Q; ++i) {
        string compRisk = classifyRisk(h_results[i]);
        if (compRisk == expected[i]) ++correct;
        
        // Print first 10 queries for reference
        if (i < 10) {
            cout << "Query " << i << ": computed=" << compRisk
                 << ", expected=" << expected[i]
                 << ", score=" << h_results[i] << "\n";
        }
    }
    
    double accuracy = Q ? (100.0 * correct / Q) : 0.0;
    cout << "\nTotal: " << Q << ", Correct: " << correct
         << ", Accuracy: " << accuracy << "%" << endl;
    cout << "Optimized PFAC kernel execution time: " << milliseconds << " ms" << endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_risk_scores));
    CUDA_CHECK(cudaFree(d_pattern_hits));

    cout.rdbuf(coutbuf); // restore old buf
    return 0;
}