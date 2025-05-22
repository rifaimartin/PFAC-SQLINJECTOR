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
#define MAX_NODES 8192    // adjust as needed
#define MAX_PATTERNS 250 // adjust as needed

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << ": " << cudaGetErrorString(err) << endl;        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Device automaton structures
__device__ int d_children[MAX_NODES * ALPHABET_SIZE];
__device__ int d_fail[MAX_NODES];
__device__ uint64_t d_maskLow[MAX_NODES];
__device__ uint64_t d_maskHigh[MAX_NODES];
__device__ int d_patternWeights[MAX_PATTERNS];

// Kernel: one thread per query
__global__ void ahoSearchKernel(
    const char *__restrict__ d_queries,
    const int *__restrict__ d_offsets,
    const int *__restrict__ d_lengths,
    int numQueries,
    int *__restrict__ d_results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numQueries)
        return;

    const char *q = d_queries + d_offsets[tid];
    int len = d_lengths[tid];
    int node = 0;
    uint64_t maskL = 0, maskH = 0;
    int risk = 0;

    for (int i = 0; i < len; ++i)
    {
        unsigned char c = (unsigned char)q[i];
        if (c >= ALPHABET_SIZE)
        {
            node = 0;
            continue;
        }
        int next = d_children[node * ALPHABET_SIZE + c];
        while (next == -1 && node != 0)
        {
            node = d_fail[node];
            next = d_children[node * ALPHABET_SIZE + c];
        }
        node = (next != -1 ? next : 0);

        // collect new patterns
        uint64_t newL = d_maskLow[node] & ~maskL;
        while (newL)
        {
            int pid = __ffsll(newL) - 1;
            risk += d_patternWeights[pid];
            newL &= newL - 1;
        }
        maskL |= d_maskLow[node];

        uint64_t newH = d_maskHigh[node] & ~maskH;
        while (newH)
        {
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
struct TrieNode
{
    unordered_map<char, TrieNode *> children;
    TrieNode *fail;
    vector<int> out; // pattern IDs
    TrieNode() : fail(nullptr) {}
};

class AhoCorasick
{
public:
    AhoCorasick()
    {
        root = new TrieNode();
        nodes.push_back(root);
    }

    void insert(const string &pat, int id)
    {
        TrieNode *u = root;
        for (char ch : pat)
        {
            if (!u->children.count(ch))
            {
                TrieNode *v = new TrieNode();
                u->children[ch] = v;
                nodes.push_back(v);
            }
            u = u->children[ch];
        }
        u->out.push_back(id);
    }

    void build()
    {
        queue<TrieNode *> q;
        root->fail = root;
        for (auto &kv : root->children)
        {
            kv.second->fail = root;
            q.push(kv.second);
        }
        while (!q.empty())
        {
            TrieNode *u = q.front();
            q.pop();
            for (auto &kv : u->children)
            {
                char ch = kv.first;
                TrieNode *v = kv.second;
                TrieNode *f = u->fail;
                while (f != root && !f->children.count(ch))
                    f = f->fail;
                if (f->children.count(ch))
                    f = f->children[ch];
                v->fail = f;
                v->out.insert(v->out.end(), f->out.begin(), f->out.end());
                q.push(v);
            }
        }
    }

    int nodeCount() const { return (int)nodes.size(); }
    int patternCount() const { return (int)patternWeights.size(); }

    TrieNode *getNode(int idx) const { return nodes[idx]; }
    int getFail(int idx) const
    {
        TrieNode *u = nodes[idx];
        return (u->fail == root ? 0 : (int)(find(nodes.begin(), nodes.end(), u->fail) - nodes.begin()));
    }
    const unordered_map<char, TrieNode *> &getChildren(int idx) const
    {
        return nodes[idx]->children;
    }
    const vector<int> &getOut(int idx) const { return nodes[idx]->out; }

    void setPatternWeights(const vector<int> &w) { patternWeights = w; }
    const vector<int> &getPatternWeights() const { return patternWeights; }

    // Expose nodes for flattening
    const vector<TrieNode *> &getNodes() const { return nodes; }

private:
    TrieNode *root;
    vector<TrieNode *> nodes;
    vector<int> patternWeights;
};

// Normalize to lowercase
string normalize(const string &s)
{
    string r = s;
    transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

// Host Classify function
string classifyRisk(int score)
{
    if (score <= 20)
        return "low"; // 0-20 (basic injection attempts)
    if (score <= 45)
        return "medium"; // 21-45 (union attacks, info disclosure)
    if (score <= 75)
        return "high"; // 46-75 (destructive operations)
    return "critical"; // 76+ (system-level attacks)
}

// Flatten automaton into host arrays
void flattenAutomaton(
    AhoCorasick &ac,
    vector<int> &h_children,
    vector<int> &h_fail,
    vector<uint64_t> &h_maskLow,
    vector<uint64_t> &h_maskHigh)
{
    int N = ac.nodeCount();
    h_children.assign(N * ALPHABET_SIZE, -1);
    h_fail.assign(N, 0);
    h_maskLow.assign(N, 0ULL);
    h_maskHigh.assign(N, 0ULL);
    const auto &nodes = ac.getNodes();

    for (int u = 0; u < N; ++u)
    {
        h_fail[u] = ac.getFail(u);
        for (auto &kv : ac.getChildren(u))
        {
            unsigned char c = kv.first;
            auto it = find(nodes.begin(), nodes.end(), kv.second);
            int v = (int)(it - nodes.begin());
            h_children[u * ALPHABET_SIZE + c] = v;
        }
        for (int pid : ac.getOut(u))
        {
            if (pid < 64)
                h_maskLow[u] |= (1ULL << pid);
            else
                h_maskHigh[u] |= (1ULL << (pid - 64));
        }
    }
}

// Read patterns from file
vector<string> readPatternsFromFile(const string &filename)
{
    vector<string> patterns;
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error: Could not open pattern file: " << filename << endl;
        return patterns;
    }

    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // Process with regex to extract patterns more reliably
    regex patternRegex("\"([^\"]+)\"");
    auto matches_begin = sregex_iterator(content.begin(), content.end(), patternRegex);
    auto matches_end = sregex_iterator();

    for (sregex_iterator i = matches_begin; i != matches_end; ++i)
    {
        smatch match = *i;
        string pattern = match[1].str(); // Get the content inside the quotes

        // Remove trailing comma if present
        if (!pattern.empty() && pattern.back() == ',')
        {
            pattern.pop_back();
        }

        // Add to patterns list
        if (!pattern.empty())
        {
            patterns.push_back(pattern);
        }
    }

    cout << "Extracted " << patterns.size() << " patterns using regex" << endl;

    if (patterns.empty())
    {
        file.open(filename);
        string line;
        while (getline(file, line))
        {

            if (line.empty() || line[0] == '#')
                continue;

            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.size() >= 2 && line.front() == '"' &&
                (line.back() == '"' || (line.size() >= 3 && line[line.size() - 2] == '"' && line.back() == ',')))
            {

                line = line.substr(1);

                // Remove closing quote and optional comma
                if (line.back() == ',')
                {
                    line = line.substr(0, line.size() - 2); // Remove both " and ,
                }
                else if (line.back() == '"')
                {
                    line = line.substr(0, line.size() - 1); // Remove just "
                }
            }
            else if (line.back() == ',')
            {
                line = line.substr(0, line.size() - 1);
            }

            if (!line.empty())
            {
                patterns.push_back(line);
            }
        }
        file.close();
        cout << "Secondary extraction found " << patterns.size() << " patterns" << endl;
    }

    return patterns;
}

void printUsage(const char *programName)
{
    cout << "Usage: " << programName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -d, --dataset <file>   CSV dataset file to process (default: sql_dataset_Critical_10000.csv)" << endl;
    cout << "  -p, --patterns <file>  Pattern file to use (default: patterns.txt)" << endl;
    cout << "  -o, --output <file>    Output file for results (default: results.txt)" << endl;
    cout << "  -h, --help             Show this help message" << endl;
}

int main(int argc, char **argv)
{
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

    ofstream out("result_AhoCorasick_" + datasetFile + ".txt");
    streambuf *coutbuf = cout.rdbuf();
    cout.rdbuf(out.rdbuf());

    vector<string> rawPatterns;

    string patternFile = "patterns.txt";

    ifstream testFile(patternFile);
    if (testFile.is_open())
    {
        testFile.close();
        rawPatterns = readPatternsFromFile(patternFile);
        if (!rawPatterns.empty())
        {
            cout << "Successfully read " << rawPatterns.size() << " patterns from " << patternFile << endl;
        }
        else
        {
            cout << "Warning: Could not extract any patterns from " << patternFile << endl;
        }
    }
    else
    {
        cout << "Warning: Could not open pattern file " << patternFile << ". Using default patterns." << endl;
    }

    int P = rawPatterns.size();

    // Display first few patterns
    cout << "First " << min(10, P) << " patterns:" << endl;
    for (int i = 0; i < min(10, P); i++)
    {
        cout << i + 1 << ". " << rawPatterns[i] << endl;
    }
    if (P > 10)
    {
        cout << "... and " << (P - 10) << " more patterns" << endl;
    }

    // Normalize patterns and assign NEW weights
    vector<string> patterns(P);
    vector<int> weights(P);

    for (int i = 0; i < P; ++i)
    {
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
            pat.find("create trigger") != string::npos)
        {
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
                 pat.find("mysql.user") != string::npos)
        {
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
                 pat.find("having") != string::npos)
        {
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
                 pat.find("limit 1") != string::npos)
        {
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
                 pat.find("'1'='1") != string::npos)
        {
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
                 pat.find("= ||") != string::npos)
        {
            weights[i] = 5;
        }

        // DEFAULT - Very basic patterns
        else
        {
            weights[i] = 1;
        }
    }

    // 2) Build host automaton
    AhoCorasick ac;
    ac.setPatternWeights(weights);
    for (int i = 0; i < P; ++i)
        ac.insert(patterns[i], i);
    ac.build();

    // 3) Flatten and copy to device
    vector<int> h_children;
    vector<int> h_fail;
    vector<uint64_t> h_maskLow, h_maskHigh;
    flattenAutomaton(ac, h_children, h_fail, h_maskLow, h_maskHigh);
    int N = ac.nodeCount();

    // Make sure we don't exceed MAX_NODES or MAX_PATTERNS
    if (N > MAX_NODES)
    {
        cerr << "Error: Automaton has " << N << " nodes, exceeding MAX_NODES (" << MAX_NODES << ")" << endl;
        cout.rdbuf(coutbuf);
        return EXIT_FAILURE;
    }
    if (P > MAX_PATTERNS)
    {
        cerr << "Error: Pattern count " << P << " exceeds MAX_PATTERNS (" << MAX_PATTERNS << ")" << endl;
        cout.rdbuf(coutbuf);
        return EXIT_FAILURE;
    }

    cout << "Automaton built with " << N << " nodes for " << P << " patterns." << endl;

    CUDA_CHECK(cudaMemcpyToSymbol(d_children, h_children.data(), N * ALPHABET_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fail, h_fail.data(), N * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_maskLow, h_maskLow.data(), N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_maskHigh, h_maskHigh.data(), N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_patternWeights, weights.data(), P * sizeof(int)));

    ifstream infile(datasetFile + ".csv");
    if (!infile.is_open())
    {
        cerr << "Error: could not open CSV file.\n";
        return EXIT_FAILURE;
    }

    string line;
    vector<string> queries;
    vector<string> expected;
    vector<string> originalQueries;

    getline(infile, line);

    int lineNum = 0;
    while (getline(infile, line))
    {
        lineNum++;
        if (line.empty())
            continue;

        vector<string> fields;
        string current_field = "";
        bool in_quotes = false;

        for (size_t i = 0; i < line.length(); i++)
        {
            char c = line[i];

            if (c == '"')
            {
                in_quotes = !in_quotes;
            }
            else if (c == ',' && !in_quotes)
            {
                fields.push_back(current_field);
                current_field = "";
            }
            else
            {
                current_field += c;
            }
        }

        fields.push_back(current_field);

        if (fields.size() >= 2)
        {
            string query_field = fields[0];
            string risk_field = fields[1];

            if (query_field.length() >= 2 && query_field.front() == '"' && query_field.back() == '"')
            {
                query_field = query_field.substr(1, query_field.length() - 2);
            }
            if (risk_field.length() >= 2 && risk_field.front() == '"' && risk_field.back() == '"')
            {
                risk_field = risk_field.substr(1, risk_field.length() - 2);
            }

            query_field.erase(0, query_field.find_first_not_of(" \t\r\n"));
            query_field.erase(query_field.find_last_not_of(" \t\r\n") + 1);
            risk_field.erase(0, risk_field.find_first_not_of(" \t\r\n"));
            risk_field.erase(risk_field.find_last_not_of(" \t\r\n") + 1);

            if (risk_field != "low" && risk_field != "medium" &&
                risk_field != "high" && risk_field != "critical")
            {
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

    // build flat buffer
    vector<int> h_offsets(Q), h_lengths(Q);
    int totalLen = 0;
    for (int i = 0; i < Q; ++i)
    {
        h_offsets[i] = totalLen;
        h_lengths[i] = queries[i].size();
        totalLen += h_lengths[i];
    }
    vector<char> h_buffer(totalLen);
    for (int i = 0; i < Q; ++i)
        memcpy(&h_buffer[h_offsets[i]], queries[i].data(), h_lengths[i]);

    // 5) Allocate device memory
    char *d_queries;
    int *d_offsets;
    int *d_lengths;
    int *d_results;
    CUDA_CHECK(cudaMalloc(&d_queries, totalLen * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_offsets, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, Q * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_queries, h_buffer.data(), totalLen * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), Q * sizeof(int), cudaMemcpyHostToDevice));

    // 6) Launch kernel
    int threads = 256;
    int blocks = (Q + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "Launching kernel with " << blocks << " blocks and " << threads << " threads per block..." << endl;

    cudaEventRecord(start);
    ahoSearchKernel<<<blocks, threads>>>(d_queries, d_offsets, d_lengths, Q, d_results);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 7) Copy back, compare, and compute accuracy
    vector<int> h_results(Q);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, Q * sizeof(int), cudaMemcpyDeviceToHost));

    int correct = 0;
    int zeroScores = 0;

    for (int i = 0; i < Q; ++i)
    {
        string compRisk = classifyRisk(h_results[i]);
        bool match = (compRisk == expected[i]);
        if (match)
            ++correct;
        if (h_results[i] == 0)
            ++zeroScores;

        // Print first 10
        if (i < 10)
        {
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
    cout << "Aho-Corasick kernel execution time: " << milliseconds << " ms" << endl;

    // cleanup
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_results));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout.rdbuf(coutbuf);

    return 0;
}