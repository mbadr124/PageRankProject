/*
 * PageRank - Message Passing Implementation (MPI)
 * Distributes nodes across MPI processes using block partitioning.
 * Each process owns a contiguous block of rows (nodes) in the CSR graph.
 * Rank values are exchanged between processes each iteration using
 * point-to-point MPI_Send / MPI_Recv.
 * Compile with: mpicxx -O2 -o pagerank_mpi pagerank_mpi.cpp
 * Run with:     mpirun -np 4 ./pagerank_mpi
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <mpi.h>

using namespace std;

// ============================================================
// CONFIGURATION
// ============================================================
const int    NUM_NODES      = 8;        // Number of nodes in the graph
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
 * Constructs a random directed graph in CSR (outgoing edges) format.
 * Called only on rank 0; the graph is then broadcast to all processes.
 * Uses a fixed seed for reproducibility across runs.
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
 * Transposes the outgoing-edge CSR to an incoming-edge CSR.
 * Each process calls this locally after receiving the full graph,
 * so that each process can compute its own node contributions
 * from the incoming-neighbor list.
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
 * pageRankMPI
 * Runs distributed PageRank using MPI point-to-point communication.
 *
 * Distribution strategy:
 *   - Nodes are partitioned into contiguous blocks across processes.
 *   - Each process updates only its assigned [local_start, local_end) nodes.
 *   - After each local update, all processes exchange their partial rank
 *     vectors using MPI_Send/MPI_Recv in a gather-broadcast pattern:
 *       1. Every non-root process sends its block to process 0.
 *       2. Process 0 sends the assembled full rank vector to all others.
 *   - Convergence is checked globally via MPI_Allreduce on the local diffs.
 *
 * Returns the full rank vector (on all processes).
 */
vector<double> pageRankMPI(const CSRGraph& g, int mpi_rank, int mpi_size) {
    int n = g.n;
    CSRGraph in = buildInCSR(g);

    // Block partition
    int block    = n / mpi_size;
    int local_start = mpi_rank * block;
    int local_end   = (mpi_rank == mpi_size - 1) ? n : local_start + block;
    int local_n     = local_end - local_start;

    vector<double> rank(n, 1.0 / n);
    vector<double> new_rank(n, 0.0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // --- Dangling sum: each process reduces its portion, then all-reduce ---
        double local_dangling = 0.0;
        for (int i = local_start; i < local_end; i++) {
            if (g.out_degree[i] == 0) local_dangling += rank[i];
        }
        double dangling_sum = 0.0;
        MPI_Allreduce(&local_dangling, &dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double base = (1.0 - DAMPING) / n + DAMPING * dangling_sum / n;

        // --- Local rank update ---
        for (int i = local_start; i < local_end; i++) {
            double contrib = 0.0;
            for (int j = in.row_ptr[i]; j < in.row_ptr[i + 1]; j++) {
                int src = in.col_idx[j];
                contrib += rank[src] / g.out_degree[src];
            }
            new_rank[i] = base + DAMPING * contrib;
        }

        // --- Gather new ranks: all non-root send to root, root broadcasts ---
        if (mpi_rank == 0) {
            // Receive blocks from all other processes
            for (int p = 1; p < mpi_size; p++) {
                int p_start = p * block;
                int p_end   = (p == mpi_size - 1) ? n : p_start + block;
                int p_n     = p_end - p_start;
                MPI_Recv(new_rank.data() + p_start, p_n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(new_rank.data() + local_start, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        // Root broadcasts the full updated rank vector
        MPI_Bcast(new_rank.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // --- Convergence check ---
        double local_diff = 0.0;
        for (int i = local_start; i < local_end; i++) {
            local_diff += abs(new_rank[i] - rank[i]);
        }
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        rank = new_rank;
        fill(new_rank.begin(), new_rank.end(), 0.0);

        if (mpi_rank == 0) {
            cout << "Iteration " << iter + 1 << " | L1 diff: " << global_diff << "\n";
        }
        if (CONVERGENCE > 0 && global_diff < CONVERGENCE) {
            if (mpi_rank == 0) {
                cout << "Converged after " << iter + 1 << " iterations.\n";
            }
            break;
        }
    }
    return rank;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    CSRGraph graph;

    // --------------------------------------------------------
    // GRAPH INIT: Choose ONE block below (comment the other)
    // --------------------------------------------------------

    // --- Option A: Random graph ---
    //if (mpi_rank == 0) graph = buildRandomGraph(NUM_NODES);

    // --- Option B: Predefined graph ---
    if (mpi_rank == 0) graph = loadGraphFromEdgeList("data/predefined_graph_edges.txt");

    // --------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();


    broadcastGraph(graph, mpi_rank);

    if (mpi_rank == 0) {
        cout << "=== MPI PageRank ===\n";
        cout << "Nodes: " << graph.n << ", Processes: " << mpi_size
                  << ", Damping: " << DAMPING << ", Max Iters: " << MAX_ITERATIONS << "\n\n";
    }

    vector<double> ranks = pageRankMPI(graph, mpi_rank, mpi_size);

    /*
    
    if (mpi_rank == 0) {
        cout << "\nFinal PageRank scores:\n";
        for (int i = 0; i < graph.n; i++) {
            cout << "  Node " << i << ": " << ranks[i] << "\n";
        }
        double sum = accumulate(ranks.begin(), ranks.end(), 0.0);
        cout << "Sum of ranks: " << sum << " (should be ~1.0)\n";
    }

    */
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if (mpi_rank == 0) {
        double elapsed_ms = (end - start) * 1000.0;
        cout << "Execution time: " << elapsed_ms << " ms\n";
    }
    MPI_Finalize();
    return 0;
}