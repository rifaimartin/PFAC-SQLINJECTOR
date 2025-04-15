// aho_cuda_full.cu
// Parallel Aho–Corasick SQLi risk detection using CUDA.

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

using namespace std;

#define ALPHABET_SIZE 128
#define MAX_NODES     8192    // adjust as needed
#define MAX_PATTERNS  256     // adjust as needed

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

// Device automaton structures
__device__ int      d_children[MAX_NODES * ALPHABET_SIZE];
__device__ int      d_fail[MAX_NODES];
__device__ uint64_t d_maskLow[MAX_NODES];
__device__ uint64_t d_maskHigh[MAX_NODES];
__device__ int      d_patternWeights[MAX_PATTERNS];

// Kernel: one thread per query
__global__ void ahoSearchKernel(
    const char* __restrict__ d_queries,
    const int*  __restrict__ d_offsets,
    const int*  __restrict__ d_lengths,
    int          numQueries,
    int*        __restrict__ d_results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numQueries) return;

    const char* q = d_queries + d_offsets[tid];
    int len = d_lengths[tid];
    int node = 0;
    uint64_t maskL = 0, maskH = 0;
    int risk = 0;

    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)q[i];
        if (c >= ALPHABET_SIZE) { node = 0; continue; }
        int next = d_children[node * ALPHABET_SIZE + c];
        while (next == -1 && node != 0) {
            node = d_fail[node];
            next = d_children[node * ALPHABET_SIZE + c];
        }
        node = (next != -1 ? next : 0);

        // collect new patterns
        uint64_t newL = d_maskLow[node] & ~maskL;
        while (newL) {
            int pid = __ffsll(newL) - 1;
            risk += d_patternWeights[pid];
            newL &= newL - 1;
        }
        maskL |= d_maskLow[node];

        uint64_t newH = d_maskHigh[node] & ~maskH;
        while (newH) {
            int pid = __ffsll(newH) - 1 + 64;
            risk += d_patternWeights[pid];
            newH &= newH - 1;
        }
        maskH |= d_maskHigh[node];
    }

    d_results[tid] = risk;
}

// ------------------------
// Host Aho–Corasick class
// ------------------------
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    TrieNode* fail;
    vector<int> out;  // pattern IDs
    TrieNode(): fail(nullptr) {}
};

class AhoCorasick {
public:
    AhoCorasick() { root = new TrieNode(); nodes.push_back(root); }

    void insert(const string& pat, int id) {
        TrieNode* u = root;
        for (char ch : pat) {
            if (!u->children.count(ch)) {
                TrieNode* v = new TrieNode();
                u->children[ch] = v;
                nodes.push_back(v);
            }
            u = u->children[ch];
        }
        u->out.push_back(id);
    }

    void build() {
        queue<TrieNode*> q;
        root->fail = root;
        for (auto &kv : root->children) {
            kv.second->fail = root;
            q.push(kv.second);
        }
        while (!q.empty()) {
            TrieNode* u = q.front(); q.pop();
            for (auto &kv : u->children) {
                char ch = kv.first;
                TrieNode* v = kv.second;
                TrieNode* f = u->fail;
                while (f != root && !f->children.count(ch)) f = f->fail;
                if (f->children.count(ch)) f = f->children[ch];
                v->fail = f;
                v->out.insert(v->out.end(), f->out.begin(), f->out.end());
                q.push(v);
            }
        }
    }

    int nodeCount()    const { return (int)nodes.size(); }
    int patternCount() const { return (int)patternWeights.size(); }

    TrieNode* getNode(int idx) const { return nodes[idx]; }
    int getFail(int idx) const {
        TrieNode* u = nodes[idx];
        return (u->fail == root ? 0 : (int)(find(nodes.begin(), nodes.end(), u->fail) - nodes.begin()));
    }
    const unordered_map<char, TrieNode*>& getChildren(int idx) const {
        return nodes[idx]->children;
    }
    const vector<int>& getOut(int idx) const { return nodes[idx]->out; }

    void setPatternWeights(const vector<int>& w) { patternWeights = w; }
    const vector<int>& getPatternWeights() const { return patternWeights; }

    // Expose nodes for flattening
    const vector<TrieNode*>& getNodes() const { return nodes; }

private:
    TrieNode* root;
    vector<TrieNode*> nodes;
    vector<int> patternWeights;
};

// Normalize to lowercase
string normalize(const string &s) {
    string r = s;
    transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

// Host classification
string classifyRisk(int score) {
    if (score <= 30)   return "low";
    if (score <= 70)   return "medium";
    if (score <= 90)   return "high";
    return "critical";
}

// Flatten automaton into host arrays
void flattenAutomaton(
    AhoCorasick& ac,
    vector<int>&      h_children,
    vector<int>&      h_fail,
    vector<uint64_t>& h_maskLow,
    vector<uint64_t>& h_maskHigh)
{
    int N = ac.nodeCount();
    h_children.assign(N * ALPHABET_SIZE, -1);
    h_fail   .assign(N, 0);
    h_maskLow.assign(N, 0ULL);
    h_maskHigh.assign(N, 0ULL);
    const auto& nodes = ac.getNodes();

    for (int u = 0; u < N; ++u) {
        h_fail[u] = ac.getFail(u);
        for (auto &kv : ac.getChildren(u)) {
            unsigned char c = kv.first;
            auto it = find(nodes.begin(), nodes.end(), kv.second);
            int v = (int)(it - nodes.begin());
            h_children[u * ALPHABET_SIZE + c] = v;
        }
        for (int pid : ac.getOut(u)) {
            if (pid < 64)
                h_maskLow[u] |= (1ULL << pid);
            else
                h_maskHigh[u] |= (1ULL << (pid - 64));
        }
    }
}

int main() {

    ofstream out("results.txt");
    streambuf* coutbuf = cout.rdbuf(); // save old buf
    cout.rdbuf(out.rdbuf());

    // 1) Define and weight patterns (expanded list)
    vector<string> rawPatterns = {
        "' or", "\" or", "' ||", "\" ||", "= or", "= ||", "' =", "' >=", "' <=", "' <>",
        "\" =", "\" !=", "= =", "= <", " >=", " <=", "' union", "' select", "' from",
        "union select", "select from", "' convert(", "' avg(", "' round(", "' sum(", "' max(", "' min(",
        ") convert(", ") avg(", ") round(", ") sum(", ") max(", ") min(", "' delete", "' drop",
        "' insert", "' truncate", "' update", "' alter", ", delete", "; drop", "; insert", "; delete", ", drop", "; truncate", "; exec", "xp_cmdshell",
        "; truncate", "' ; update", "like or", "like ||", "' %", "like %", " %", "</script>", "</script >",
        "union", "select", "drop", "insert", "delete", "update", "or 1=1", "--", "#", "/*", "*/",
        "sleep(", "benchmark(", "count(*)", "information_schema.schemata", "null", "version(", "current_user",
        "outfile", "load_file"
    };
    int P = rawPatterns.size();
    vector<string> patterns(P);
    vector<int> weights(P);
    for (int i = 0; i < P; ++i) {
        patterns[i] = normalize(rawPatterns[i]);
        const auto &pat = patterns[i];
        if (pat.find("; drop")!=string::npos || pat.find("xp_cmdshell")!=string::npos ||
            pat.find("; exec")!=string::npos || pat.find("outfile")!=string::npos ||
            pat.find("load_file")!=string::npos)
            weights[i] = 100;
        else if (pat.find("; delete")!=string::npos || pat.find("; insert")!=string::npos ||
                 pat.find("; truncate")!=string::npos || pat.find("; update")!=string::npos ||
                 pat.find("sleep(")!=string::npos || pat.find("version(")!=string::npos ||
                 pat.find("current_user")!=string::npos)
            weights[i] = 15;
        else
            weights[i] = 10;
    }

    // 2) Build host automaton
    AhoCorasick ac;
    ac.setPatternWeights(weights);
    for (int i = 0; i < P; ++i) ac.insert(patterns[i], i);
    ac.build();

    // 3) Flatten and copy to device
    vector<int>      h_children;
    vector<int>      h_fail;
    vector<uint64_t> h_maskLow, h_maskHigh;
    flattenAutomaton(ac, h_children, h_fail, h_maskLow, h_maskHigh);
    int N = ac.nodeCount();
    CUDA_CHECK(cudaMemcpyToSymbol(d_children, h_children.data(), N * ALPHABET_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fail,     h_fail.data(),     N * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_maskLow,  h_maskLow.data(),  N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_maskHigh, h_maskHigh.data(), N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_patternWeights, weights.data(), P * sizeof(int)));

    // 4) Read queries and expected risks
    ifstream infile("sqli_dataset_Low_New.csv");
    if (!infile.is_open()) {
        cerr << "Error: could not open CSV file.\n";
        return EXIT_FAILURE;
    }
    string line;
    vector<string> queries;
    vector<string> expected;
    // skip header
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

    // build flat buffer
    vector<int>  h_offsets(Q), h_lengths(Q);
    int totalLen = 0;
    for (int i = 0; i < Q; ++i) {
        h_offsets[i] = totalLen;
        h_lengths[i] = queries[i].size();
        totalLen += h_lengths[i];
    }
    vector<char> h_buffer(totalLen);
    for (int i = 0; i < Q; ++i)
        memcpy(&h_buffer[h_offsets[i]], queries[i].data(), h_lengths[i]);

    // 5) Allocate device memory
    char* d_queries;   int* d_offsets;   int* d_lengths;   int* d_results;
    CUDA_CHECK(cudaMalloc(&d_queries, totalLen * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_offsets, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, Q * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_queries, h_buffer.data(), totalLen * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), Q * sizeof(int), cudaMemcpyHostToDevice));

    // 6) Launch kernel
    int threads = 256;
    int blocks  = (Q + threads - 1) / threads;
    ahoSearchKernel<<<blocks, threads>>>(d_queries, d_offsets, d_lengths, Q, d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7) Copy back, compare, and compute accuracy
    vector<int> h_results(Q);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, Q * sizeof(int), cudaMemcpyDeviceToHost));

    int correct = 0;
    for (int i = 0; i < Q; ++i) {
        string compRisk = classifyRisk(h_results[i]);
        bool match = (compRisk == expected[i]);
        if (match) ++correct;
        cout << "Query " << i << ": computed=" << compRisk
                  << ", expected=" << expected[i]
                  << (match ? " [OK]" : " [Mismatch]") << "\n";
    }
    double accuracy = Q ? (100.0 * correct / Q) : 0.0;
    cout << "\nTotal: " << Q << ", Correct: " << correct
              << ", Accuracy: " << accuracy << "%\n";

    // cleanup
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_results));

    cout.rdbuf(coutbuf); // restore old buf if you want to print to console again

    return 0;
}
