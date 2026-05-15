#include <iostream>
#include <cmath>
#include <numeric>

#include "pagerank_common.h"

using namespace std;

vector<double> pageRank(const CSRGraph& g, double* elapsed_ms_out) {
    int n = g.n;
    CSRGraph in = buildInCSR(g);
    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

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

    clock_gettime(CLOCK_MONOTONIC, &end);
    if (elapsed_ms_out) {
        *elapsed_ms_out = (end.tv_sec - start.tv_sec) * 1000.0 +
                          (end.tv_nsec - start.tv_nsec) / 1e6;
    }
    return rank;
}

int main() {
    string graph_file = "data/predefined_graph_edges.txt";
    cout << "Loading graph from: " << graph_file << "\n";
    // Use a predefined graph from data/.
    CSRGraph graph = loadGraphFromEdgeList(graph_file);

    cout << "=== Sequential PageRank ===\n";
    cout << "Nodes: " << graph.n << ", Damping: " << DAMPING
              << ", Max Iters: " << MAX_ITERATIONS << "\n\n";

    double elapsed_ms = 0.0;
    vector<double> ranks = pageRank(graph, &elapsed_ms);

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