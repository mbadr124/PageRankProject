#ifndef PAGERANK_COMMON_H
#define PAGERANK_COMMON_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>

using namespace std;

// CONFIGURATION
constexpr int MAX_ITERATIONS = 50000;  // Maximum number of iterations
constexpr double CONVERGENCE = 1e-6;   // Convergence threshold
constexpr double DAMPING = 0.85;       // Damping factor
constexpr const char* GRAPH_FILE = "data/graph_2k_edges.txt";

// CSR Graph Structure
struct CSRGraph {
    int n;                          // Number of nodes
    vector<int> row_ptr;       // row_ptr[i]..row_ptr[i+1]-1 = outgoing neighbors of node i
    vector<int> col_idx;       // Destination node indices
    vector<int> out_degree;    // Out-degree of each node
};

// Common graph helpers
inline CSRGraph loadGraphFromEdgeList(const string& filename) {
    ifstream file(filename);
    if (!file) {
        throw runtime_error("Could not open graph file: " + filename);
    }

    int n = 0;
    int m = 0;
    file >> n >> m;
    if (!file || n <= 0 || m < 0) {
        throw runtime_error("Invalid graph header in: " + filename);
    }

    vector<vector<int>> adj(n);
    for (int edge = 0; edge < m; edge++) {
        int src = 0;
        int dst = 0;
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
        g.out_degree[i] = static_cast<int>(adj[i].size());
        g.row_ptr[i + 1] = g.row_ptr[i] + g.out_degree[i];
        for (int dst : adj[i]) {
            g.col_idx.push_back(dst);
        }
    }
    return g;
}

inline CSRGraph buildInCSR(const CSRGraph& g) {
    int n = g.n;
    CSRGraph in;
    in.n = n;
    in.row_ptr.resize(n + 1, 0);

    in.out_degree.resize(n, 0);

    for (int dst : g.col_idx) {
        in.row_ptr[dst + 1]++;
    }
    for (int i = 1; i <= n; i++) {
        in.row_ptr[i] += in.row_ptr[i - 1];
    }
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


// Broadcast CSR graph from process 0 to all MPI processes
inline void broadcastGraph(CSRGraph& graph, int mpi_rank) {

    
    // Broadcast number of nodes
    
    MPI_Bcast(
        &graph.n,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    int row_ptr_size;
    int col_idx_size;
    int out_degree_size;

    
    // Process 0 provides vector sizes
    
    if (mpi_rank == 0) {

        row_ptr_size =
            static_cast<int>(graph.row_ptr.size());

        col_idx_size =
            static_cast<int>(graph.col_idx.size());

        out_degree_size =
            static_cast<int>(graph.out_degree.size());
    }

    
    // Broadcast vector sizes
    
    MPI_Bcast(
        &row_ptr_size,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        &col_idx_size,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        &out_degree_size,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    
    // Allocate memory on non-root processes
    
    if (mpi_rank != 0) {

        graph.row_ptr.resize(row_ptr_size);

        graph.col_idx.resize(col_idx_size);

        graph.out_degree.resize(out_degree_size);
    }

    
    // Broadcast CSR arrays
    
    MPI_Bcast(
        graph.row_ptr.data(),
        row_ptr_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        graph.col_idx.data(),
        col_idx_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        graph.out_degree.data(),
        out_degree_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
}

#endif