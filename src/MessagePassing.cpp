/*
 * PageRank - Message Passing Implementation (MPI)
 * Distributes nodes across MPI processes using block partitioning.
 *
 * Compile with:
 * mpicxx -O2 -o pagerank_mpi pagerank_mpi.cpp
 *
 * Run with:
 * mpirun -np 4 ./pagerank_mpi
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
 * Runs distributed PageRank using MPI point-to-point communication.
 *
 * Distribution strategy:
 * - Nodes are partitioned into contiguous blocks.
 * - Each process updates only its assigned nodes.
 * - Partial rank vectors are gathered on root.
 * - Root broadcasts the updated global vector.
 * - Global convergence uses MPI_Allreduce.
 */
vector<double> pageRankMPI(
    const CSRGraph& g,
    int mpi_rank,
    int mpi_size,
    double* elapsed_ms_out
) {

    int n = g.n;

    CSRGraph in = buildInCSR(g);

    // --------------------------------------------------------
    // Block partitioning
    // --------------------------------------------------------
    int block = n / mpi_size;

    int local_start = mpi_rank * block;

    int local_end =
        (mpi_rank == mpi_size - 1)
        ? n
        : local_start + block;

    int local_n = local_end - local_start;

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    // --------------------------------------------------------
    // Start timing
    // --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        // ----------------------------------------------------
        // Dangling-node contribution
        // ----------------------------------------------------
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

        // ----------------------------------------------------
        // Local rank update
        // ----------------------------------------------------
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

        // ----------------------------------------------------
        // Gather updated ranks on root
        // ----------------------------------------------------
        if (mpi_rank == 0) {

            // Root already owns its own block

            for (int p = 1; p < mpi_size; p++) {

                int p_start = p * block;

                int p_end =
                    (p == mpi_size - 1)
                    ? n
                    : p_start + block;

                int p_n = p_end - p_start;

                MPI_Recv(
                    new_rank.data() + p_start,
                    p_n,
                    MPI_DOUBLE,
                    p,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
            }

        } else {

            MPI_Send(
                new_rank.data() + local_start,
                local_n,
                MPI_DOUBLE,
                0,
                0,
                MPI_COMM_WORLD
            );
        }

        // ----------------------------------------------------
        // Broadcast full updated vector
        // ----------------------------------------------------
        MPI_Bcast(
            new_rank.data(),
            n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // ----------------------------------------------------
        // Convergence check
        // ----------------------------------------------------
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

    // --------------------------------------------------------
    // End timing
    // --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (elapsed_ms_out && mpi_rank == 0) {

        *elapsed_ms_out =
            (end - start) * 1000.0;
    }

    return rank;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int mpi_rank;
    int mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // --------------------------------------------------------
    // Load graph
    // --------------------------------------------------------
    CSRGraph graph;

    string graph_file =
        "data/predefined_graph_edges.txt";

    if (mpi_rank == 0) {

        cout << "Loading graph from: "
             << graph_file
             << "\n";

        graph =
            loadGraphFromEdgeList(graph_file);
    }

    // Broadcast graph to all processes
    broadcastGraph(graph, mpi_rank);

    // --------------------------------------------------------

    if (mpi_rank == 0) {

        cout << "=== MPI PageRank ===\n";

        cout << "Nodes: " << graph.n
             << ", Processes: " << mpi_size
             << ", Damping: " << DAMPING
             << ", Max Iters: " << MAX_ITERATIONS
             << "\n\n";
    }

    double elapsed_ms = 0.0;

    vector<double> ranks =
        pageRankMPI(
            graph,
            mpi_rank,
            mpi_size,
            &elapsed_ms
        );

    /*
    if (mpi_rank == 0) {

        cout << "\nFinal PageRank scores:\n";

        for (int i = 0; i < graph.n; i++) {

            cout << "Node "
                 << i
                 << ": "
                 << ranks[i]
                 << "\n";
        }
    }
    */

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