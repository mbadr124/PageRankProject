/*
 * PageRank - Shared Memory Implementation (OpenMP)
 * Parallelizes rank updates across threads using OpenMP.
 * Uses reductions for dangling node sums and atomic-free updates
 * via double-buffering (read from rank[], write to new_rank[]).
 * Compile with: g++ -O2 -fopenmp -o pagerank_omp pagerank_omp.cpp
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <omp.h>

using namespace std;

// ============================================================
// CONFIGURATION
// ============================================================
const int    NUM_NODES      = 8;        // Number of nodes in the graph
const int    NUM_THREADS    = 4;        // Number of OpenMP threads
const int    MAX_ITERATIONS = 50000;      // Maximum number of iterations
const double CONVERGENCE    = 1e-6;     // Convergence threshold (set to -1 to always run MAX_ITERATIONS)
const double DAMPING        = 0.85;     // Damping factor

// ============================================================
// CSR Graph Structure
// ============================================================
struct CSRGraph {
    int n;
    vector<int> row_ptr;
    vector<int> col_idx;
    vector<int> out_degree;
};

/*
 * buildRandomGraph
 * Constructs a random directed graph in CSR format.
 * Each node gets a random number of outgoing edges to other nodes.
 * Used when the user wants to test with a randomly generated graph.
 */
CSRGraph buildRandomGraph(int n) {
    srand(42);
    CSRGraph g;
    g.n = n;
    g.row_ptr.resize(n + 1, 0);
    g.out_degree.resize(n, 0);

    vector<vector<int>> adj(n);
    for (int i = 0; i < n; i++) {
        int deg = rand() % (n - 1) + 1;
        for (int d = 0; d < deg; d++) {
            int dst = rand() % n;
            if (dst != i) adj[i].push_back(dst);
        }
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
        g.out_degree[i] = (int)adj[i].size();
    }

    for (int i = 0; i < n; i++) {
        g.row_ptr[i + 1] = g.row_ptr[i] + (int)adj[i].size();
        for (int dst : adj[i]) g.col_idx.push_back(dst);
    }
    return g;
}

/*
 * loadGraphFromEdgeList
 * Reads a directed graph from a plain text edge-list file and stores it in CSR format.
 * File format:
 *   num_nodes num_edges
 *   source_node destination_node
 *   source_node destination_node
 */
CSRGraph loadGraphFromEdgeList(const string& filename) {
    ifstream file(filename);
    if (!file) {
        throw runtime_error("Could not open graph file: " + filename);
    }

    int n, m;
    file >> n >> m;
    if (!file || n <= 0 || m < 0) {
        throw runtime_error("Invalid graph header in: " + filename);
    }

    vector<vector<int>> adj(n);
    for (int edge = 0; edge < m; edge++) {
        int src, dst;
        file >> src >> dst;
        if (!file) {
            throw runtime_error("Invalid edge list data in: " + filename);
        }
        if (src < 0 || src >= n || dst < 0 || dst >= n) {
            throw runtime_error("Graph edge has a node outside the valid range in: " + filename);
        }
        adj[src].push_back(dst);
    }

    CSRGraph g;
    g.n = n;
    g.row_ptr.resize(n + 1, 0);
    g.out_degree.resize(n, 0);

    for (int i = 0; i < n; i++) {
        g.out_degree[i] = (int)adj[i].size();
        g.row_ptr[i + 1] = g.row_ptr[i] + g.out_degree[i];
        for (int dst : adj[i]) g.col_idx.push_back(dst);
    }
    return g;
}

/*
 * buildInCSR
 * Converts an outgoing-edge CSR graph into an incoming-edge CSR.
 * PageRank sums contributions from incoming neighbors,
 * so we transpose the graph for efficient per-node traversal.
 */
CSRGraph buildInCSR(const CSRGraph& g) {
    int n = g.n;
    CSRGraph in;
    in.n = n;
    in.row_ptr.resize(n + 1, 0);
    in.out_degree.resize(n, 0);

    for (int dst : g.col_idx) in.row_ptr[dst + 1]++;
    for (int i = 1; i <= n; i++) in.row_ptr[i] += in.row_ptr[i - 1];
    in.col_idx.resize(g.col_idx.size());

    vector<int> pos(in.row_ptr.begin(), in.row_ptr.end());
    for (int src = 0; src < n; src++) {
        for (int j = g.row_ptr[src]; j < g.row_ptr[src + 1]; j++) {
            int dst = g.col_idx[j];
            in.col_idx[pos[dst]++] = src;
        }
    }
    return in;
}

/*
 * pageRankOMP
 * Runs PageRank using OpenMP for parallelism.
 *
 * Race-condition safety strategy:
 *   - Double-buffering: each thread reads only from rank[] and writes
 *     only to its own new_rank[i] slot — no two threads write the same index.
 *     This eliminates race conditions without requiring atomic operations.
 *   - Dangling-node sum uses an OpenMP reduction clause, which safely
 *     accumulates partial sums from each thread into a single scalar.
 *   - Convergence diff also uses a reduction over the parallel loop.
 *
 * Returns the final rank vector.
 */
vector<double> pageRankOMP(const CSRGraph& g) {
    int n = g.n;
    CSRGraph in = buildInCSR(g);

    omp_set_num_threads(NUM_THREADS);

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // --- Dangling node sum (reduction) ---
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum) schedule(static)
        for (int i = 0; i < n; i++) {
            if (g.out_degree[i] == 0) dangling_sum += rank[i];
        }

        double base = (1.0 - DAMPING) / n + DAMPING * dangling_sum / n;

        // --- Parallel rank update (no race: each thread owns its i) ---
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            double contrib = 0.0;
            for (int j = in.row_ptr[i]; j < in.row_ptr[i + 1]; j++) {
                int src = in.col_idx[j];
                contrib += rank[src] / g.out_degree[src];
            }
            new_rank[i] = base + DAMPING * contrib;
        }

        // --- Convergence check (reduction) ---
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff) schedule(static)
        for (int i = 0; i < n; i++) {
            diff += abs(new_rank[i] - rank[i]);
            rank[i] = new_rank[i];
            new_rank[i] = 0.0;
        }

        cout << "Iteration " << iter + 1 << " | L1 diff: " << diff << "\n";
        if (CONVERGENCE > 0 && diff < CONVERGENCE) {
            cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
    }
    return rank;
}

int main() {
    // --------------------------------------------------------
    // GRAPH INIT: Choose ONE block below (comment the other)
    // --------------------------------------------------------

    // --- Option A: Random graph ---
    //CSRGraph graph = buildRandomGraph(NUM_NODES);

    // --- Option B: Predefined graph ---
    CSRGraph graph = loadGraphFromEdgeList("data/predefined_graph_edges.txt");

    // --------------------------------------------------------

    cout << "=== OpenMP PageRank ===\n";
    cout << "Nodes: " << graph.n << ", Threads: " << NUM_THREADS
              << ", Damping: " << DAMPING << ", Max Iters: " << MAX_ITERATIONS << "\n\n";

    double start = omp_get_wtime();
    
    vector<double> ranks = pageRankOMP(graph);

    double end = omp_get_wtime();              
    
    /*
    
    cout << "\nFinal PageRank scores:\n";
    for (int i = 0; i < graph.n; i++) {
        cout << "  Node " << i << ": " << ranks[i] << "\n";
    }

    */
   
    double sum = accumulate(ranks.begin(), ranks.end(), 0.0);
    cout << "Sum of ranks: " << sum << " (should be ~1.0)\n";
    double elapsed_ms = (end - start) * 1000.0;
    cout << "Execution time: " << elapsed_ms << " ms\n";
    return 0;
}