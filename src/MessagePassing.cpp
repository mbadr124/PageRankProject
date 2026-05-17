/*
 * PageRank - Message Passing Implementation (MPI)
 * Distributes nodes across MPI processes using block partitioning.
 *
 * Compile with:
cmd /c "chcp 65001>nul & C:\msys64\ucrt64\bin\g++.exe -O2 -DNDEBUG -fdiagnostics-color=always C:\Users\MohammedBadr\Desktop\HPC\Project\pagerank-project\src\MessagePassing.cpp -I C:\MPI\SDK\Include\ -L C:\MPI\SDK\Lib\x64\ -lmsmpi -o C:\Users\MohammedBadr\Desktop\HPC\Project\pagerank-project\bin\pagerank_mpi.exe" 
* Run with:
 *   mpiexec -n 4 .\bin\pagerank_mpi.exe
 */

#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <mpi.h>

#include "pagerank_common.h"

using namespace std;

/*
 * pageRankMPI
 * Runs distributed PageRank using MPI collective communication.
 *
 * Distribution strategy:
 * - Nodes are partitioned into contiguous blocks.
 * - Each process updates only its assigned nodes.
 * - Updated partial rank vectors are combined with MPI_Allgatherv.
 * - Global convergence uses MPI_Allreduce.
 */
vector<double> pageRankMPI(
    const CSRGraph& g,
    int mpi_rank,
    int mpi_size
) {

    int n = g.n;

    CSRGraph in = buildInCSR(g);

    // Block partitioning
    int block = n / mpi_size;

    int local_start = mpi_rank * block;

    int local_end =
        (mpi_rank == mpi_size - 1)
        ? n
        : local_start + block;

    int local_n = local_end - local_start;

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    vector<int> counts(mpi_size, 0);
    vector<int> displs(mpi_size, 0);

    for (int p = 0; p < mpi_size; p++) {

        int p_start = p * block;

        int p_end =
            (p == mpi_size - 1)
            ? n
            : p_start + block;

        counts[p] = p_end - p_start;
        displs[p] = p_start;
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // Dangling-node contribution
        double local_dangling = 0.0;

        for (int i = local_start; i < local_end; i++) {

            if (g.out_degree[i] == 0) {
                local_dangling += rank[i];
            }
        }

        double dangling_sum = 0.0;

        MPI_Allreduce(
            &local_dangling,
            &dangling_sum,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        double base =
            (1.0 - DAMPING) / n +
            DAMPING * dangling_sum / n;

        // Local rank update
        for (int i = local_start; i < local_end; i++) {

            double contrib = 0.0;

            for (int j = in.row_ptr[i];
                 j < in.row_ptr[i + 1];
                 j++) {

                int src = in.col_idx[j];

                contrib += rank[src] / g.out_degree[src];
            }

            new_rank[i] = base + DAMPING * contrib;
        }

        // Exchange updated ranks across all processes
        MPI_Allgatherv(
            new_rank.data() + local_start,
            local_n,
            MPI_DOUBLE,
            new_rank.data(),
            counts.data(),
            displs.data(),
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Convergence check
        double local_diff = 0.0;

        for (int i = local_start; i < local_end; i++) {

            local_diff += abs(new_rank[i] - rank[i]);
        }

        double global_diff = 0.0;

        MPI_Allreduce(
            &local_diff,
            &global_diff,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        rank = new_rank;

        fill(new_rank.begin(), new_rank.end(), 0.0);

        if (mpi_rank == 0) {

            cout << "Iteration "
                 << iter + 1
                 << " | L1 diff: "
                 << global_diff
                 << "\n";
        }

        if (CONVERGENCE > 0 &&
            global_diff < CONVERGENCE) {

            if (mpi_rank == 0) {

                cout << "Converged after "
                     << iter + 1
                     << " iterations.\n";
            }

            break;
        }
    }

    return rank;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int mpi_rank;
    int mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Load graph
    CSRGraph graph;

    string graph_file = GRAPH_FILE;

    if (mpi_rank == 0) {

        cout << "Loading graph from: "
             << graph_file
             << "\n";

        graph =
            loadGraphFromEdgeList(graph_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Broadcast graph to all processes
    broadcastGraph(graph, mpi_rank);

    if (mpi_rank == 0) {

        cout << "=== MPI PageRank ===\n";

        cout << "Nodes: " << graph.n
             << ", Processes: " << mpi_size
             << ", Damping: " << DAMPING
             << ", Max Iters: " << MAX_ITERATIONS
             << "\n\n";
    }

    vector<double> ranks =
        pageRankMPI(
            graph,
            mpi_rank,
            mpi_size
        );

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed_ms = (t_end - t_start) * 1000.0;


    if (mpi_rank == 0) {

        double sum =
            accumulate(
                ranks.begin(),
                ranks.end(),
                0.0
            );

        cout << "Sum of ranks: "
             << sum
             << " (should be ~1.0)\n";

        cout << "Execution time: "
             << elapsed_ms
             << " ms\n";
    }

    MPI_Finalize();

    return 0;
}