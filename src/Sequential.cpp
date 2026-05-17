/*
 * Sequential PageRank implementation in C++.
 *
 * This version computes PageRank scores in a single thread, iterating until convergence or a maximum number of iterations.
 *
 * Compile with:
  g++ -O2 -I"$env:MSMPI_INC" -L"$env:MSMPI_LIB64" -lmsmpi -o bin/pagerank_seq src/Sequential.cpp
 *
 * Note: Ensure you have the graph data file at "data/predefined_graph_edges.txt" for testing.
 */

#include <iostream>
#include <cmath>
#include <numeric>

#include "pagerank_common.h"

using namespace std;

vector<double> pageRank(const CSRGraph& g) {
    int n = g.n;
    CSRGraph in = buildInCSR(g);
    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Handle dangling nodes (nodes with no outgoing edges)
        double dangling_sum = 0.0;
        for (int i = 0; i < n; i++) {
            if (g.out_degree[i] == 0) dangling_sum += rank[i];
        }

        double base = (1.0 - DAMPING) / n + DAMPING * dangling_sum / n; 

        for (int i = 0; i < n; i++) {
            double contrib = 0.0;
            for (int j = in.row_ptr[i]; j < in.row_ptr[i + 1]; j++) {
                int src = in.col_idx[j];
                contrib += rank[src] / g.out_degree[src];
            }
            new_rank[i] = base + DAMPING * contrib;
        }

        // Check convergence
        double diff = 0.0;
        for (int i = 0; i < n; i++) diff += abs(new_rank[i] - rank[i]);
        rank = new_rank;
        fill(new_rank.begin(), new_rank.end(), 0.0);

        cout << "Iteration " << iter + 1 << " | L1 diff: " << diff << "\n";
        if (CONVERGENCE > 0 && diff < CONVERGENCE) {
            cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
    }

    return rank;
}

int main() {
    string graph_file = GRAPH_FILE;

    cout << "Loading graph from: " << graph_file << "\n";

    // Use a predefined graph from data/.
    CSRGraph graph = loadGraphFromEdgeList(graph_file);

    cout << "=== Sequential PageRank ===\n";
    cout << "Nodes: " << graph.n << ", Damping: " << DAMPING
              << ", Max Iters: " << MAX_ITERATIONS << "\n\n";

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    vector<double> ranks = pageRank(graph);
    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e6;

    /*
    
                        cout << "\nFinal PageRank scores:\n";
    for (int i = 0; i < graph.n; i++) {
        cout << "  Node " << i << ": " << ranks[i] << "\n";
    }

    */

    double sum = accumulate(ranks.begin(), ranks.end(), 0.0);
    cout << "Sum of ranks: " << sum << " (should be ~1.0)\n";
    cout << "Execution time: " << elapsed_ms << " ms\n";
    return 0;
}