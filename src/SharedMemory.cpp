/*
 * PageRank - Shared Memory Implementation (OpenMP)
 * Parallelizes rank updates across threads using OpenMP.
 *
 * Compile with:
  g++ -O2 -fopenmp -I"$env:MSMPI_INC" -L"$env:MSMPI_LIB64" -lmsmpi -o bin/pagerank_omp src/SharedMemory.cpp
 */

#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <omp.h>

#include "pagerank_common.h"

using namespace std;

// CONFIGURATION
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
vector<double> pageRankOMP(const CSRGraph& g, int num_threads) {

    int n = g.n;

    // Build incoming-edge CSR representation
    CSRGraph in = buildInCSR(g);

    omp_set_num_threads(num_threads);

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // Compute dangling-node contribution
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

        // Parallel rank update
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

        // Convergence check
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

    return rank;
}

int main(int argc, char** argv) {

    // Load graph from edge list
    string graph_file = GRAPH_FILE;

    cout << "Loading graph from: "
         << graph_file
         << "\n";

    CSRGraph graph =
        loadGraphFromEdgeList(graph_file);


    int num_threads = NUM_THREADS;
    if (argc >= 2) {
        try {
            num_threads = stoi(argv[1]);
        } catch (const exception&) {
            cerr << "Invalid thread count: " << argv[1] << "\n";
            return 1;
        }
        if (num_threads <= 0) {
            cerr << "Thread count must be positive.\n";
            return 1;
        }
    }

    cout << "=== OpenMP PageRank ===\n";

    cout << "Nodes: " << graph.n
         << ", Threads: " << num_threads
         << ", Damping: " << DAMPING
         << ", Max Iters: " << MAX_ITERATIONS
         << "\n\n";

    double t_start = omp_get_wtime();

    vector<double> ranks =
        pageRankOMP(graph, num_threads);

    double t_end = omp_get_wtime();

    double elapsed_ms = (t_end - t_start) * 1000.0;


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