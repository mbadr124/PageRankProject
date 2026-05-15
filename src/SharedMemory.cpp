/*
 * PageRank - Shared Memory Implementation (OpenMP)
 * Parallelizes rank updates across threads using OpenMP.
 *
 * Compile with:
 * g++ -O2 -fopenmp -o pagerank_omp pagerank_omp.cpp
 */

#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <omp.h>

#include "pagerank_common.h"

using namespace std;

// ============================================================
// CONFIGURATION
// ============================================================
const int NUM_THREADS = 4;

/*
 * pageRankOMP
 * Runs PageRank using OpenMP parallelism.
 *
 * Parallelization strategy:
 * - Double buffering:
 *     Reads from rank[]
 *     Writes to new_rank[]
 *   This avoids race conditions.
 *
 * - OpenMP reductions:
 *     Used for dangling node accumulation
 *     Used for convergence difference accumulation
 */
vector<double> pageRankOMP(
    const CSRGraph& g,
    double* elapsed_ms_out
) {

    int n = g.n;

    // Build incoming-edge CSR representation
    CSRGraph in = buildInCSR(g);

    omp_set_num_threads(NUM_THREADS);

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    // --------------------------------------------------------
    // Start timing (measure iterative computation only)
    // --------------------------------------------------------
    double start = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // ----------------------------------------------------
        // Compute dangling-node contribution
        // ----------------------------------------------------
        double dangling_sum = 0.0;

        #pragma omp parallel for reduction(+:dangling_sum) schedule(static)
        for (int i = 0; i < n; i++) {

            if (g.out_degree[i] == 0) {
                dangling_sum += rank[i];
            }
        }

        double base =
            (1.0 - DAMPING) / n +
            DAMPING * dangling_sum / n;

        // ----------------------------------------------------
        // Parallel rank update
        // ----------------------------------------------------
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {

            double contrib = 0.0;

            for (int j = in.row_ptr[i];
                 j < in.row_ptr[i + 1];
                 j++) {

                int src = in.col_idx[j];

                contrib += rank[src] / g.out_degree[src];
            }

            new_rank[i] = base + DAMPING * contrib;
        }

        // ----------------------------------------------------
        // Convergence check
        // ----------------------------------------------------
        double diff = 0.0;

        #pragma omp parallel for reduction(+:diff) schedule(static)
        for (int i = 0; i < n; i++) {

            diff += abs(new_rank[i] - rank[i]);

            rank[i] = new_rank[i];
            new_rank[i] = 0.0;
        }

        cout << "Iteration "
             << iter + 1
             << " | L1 diff: "
             << diff
             << "\n";

        if (CONVERGENCE > 0 && diff < CONVERGENCE) {

            cout << "Converged after "
                 << iter + 1
                 << " iterations.\n";

            break;
        }
    }

    // --------------------------------------------------------
    // End timing
    // --------------------------------------------------------
    double end = omp_get_wtime();

    if (elapsed_ms_out) {
        *elapsed_ms_out =
            (end - start) * 1000.0;
    }

    return rank;
}

int main() {

    // --------------------------------------------------------
    // Load graph from edge list
    // --------------------------------------------------------
    string graph_file =
        "data/predefined_graph_edges.txt";

    cout << "Loading graph from: "
         << graph_file
         << "\n";

    CSRGraph graph =
        loadGraphFromEdgeList(graph_file);

    // --------------------------------------------------------

    cout << "=== OpenMP PageRank ===\n";

    cout << "Nodes: " << graph.n
         << ", Threads: " << NUM_THREADS
         << ", Damping: " << DAMPING
         << ", Max Iters: " << MAX_ITERATIONS
         << "\n\n";

    double elapsed_ms = 0.0;

    vector<double> ranks =
        pageRankOMP(graph, &elapsed_ms);

    // --------------------------------------------------------
    // Optional: Print final ranks
    // --------------------------------------------------------
    /*
    cout << "\nFinal PageRank scores:\n";

    for (int i = 0; i < graph.n; i++) {

        cout << "Node "
             << i
             << ": "
             << ranks[i]
             << "\n";
    }
    */

    double sum =
        accumulate(ranks.begin(), ranks.end(), 0.0);

    cout << "Sum of ranks: "
         << sum
         << " (should be ~1.0)\n";

    cout << "Execution time: "
         << elapsed_ms
         << " ms\n";

    return 0;
}