#include <iostream>
#include <vector>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "ECL_GRAPH.h"

#define NUM_NODES 50'000'005
#define NUM_EDGES 300'000'005

using namespace std;

// Struct to represent a node in the graph
typedef struct{
	int start;     // Index of first adjacent node in edges_ptr	
	int length;    // Number of adjacent nodes 
} Node;

// Global variables for the graph and BFS traversal
Node node[NUM_NODES];
int edges[NUM_EDGES], source = 1, num_nodes, num_edges;
bool curr_frontier[NUM_NODES] = {false};
bool next_frontier[NUM_NODES] = {false};
bool visited[NUM_NODES] = {false};

// Device pointers for CUDA kernel
int* num_nodes_ptr;
int* edges_ptr; // CSR edges
Node* node_ptr; // CSR start and lengths
bool* currqueue, *nextqueue, *visited_ptr, *done;

// CUDA kernel to perform BFS traversal
__global__ void BFS_KERNEL(Node* node_ptr, int* edges_ptr, bool* currqueue, bool* nextqueue, 
    bool* visited_ptr, bool* done, int* num_nodes_ptr){
	// Each thread computes the BFS traversal for a single node
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Ensure the thread's id is within the valid range and the current node is in the current frontier
	if (id < *num_nodes_ptr && currqueue[id] == true){
		// Mark the current node as visited and remove it from the current frontier
		visited_ptr[id] = true;
		currqueue[id] = false;

		// Get the starting index and ending index of the edges associated with this node
		int start = node_ptr[id].start;
		int end = start + node_ptr[id].length;

		// Traverse all the edges of the current node
		for (int i = start; i < end; i++){
			// Get the id of the neighbor node
			int nid = edges_ptr[i];
			// Check if the neighbor node is not already visited and not in the current frontier
			if (!currqueue[nid] && !visited_ptr[nid]){
				// Add the neighbor node to the next frontier
				nextqueue[nid] = true;
				// Set the 'done' flag to false to indicate that the BFS is not yet complete
				*done = false;
			}
		}
	}
}

int NUMBER_OF_BLOCKS, NUMBER_OF_THREADS;

// Function to perform BFS traversal on the graph
void bfs_caller(){
	// Start timer for preprocessing
	auto start_pp = std::chrono::high_resolution_clock::now();

	int count = 0;
	bool done_val = true;

	// Initialize the source node in the current frontier
	curr_frontier[source] = true;

	// Allocate device memory and copy data to device
	cudaMalloc((void**)&num_nodes_ptr, sizeof(int));
	cudaMemcpy(num_nodes_ptr, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&node_ptr, sizeof(Node) * num_nodes);
	cudaMemcpy(node_ptr, node, sizeof(Node) * num_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&edges_ptr, sizeof(int) * num_edges);
	cudaMemcpy(edges_ptr, edges, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&currqueue, sizeof(bool) * num_nodes);
	cudaMemcpy(currqueue, curr_frontier, sizeof(bool) * num_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&nextqueue, sizeof(bool) * num_nodes);
	cudaMemcpy(nextqueue, next_frontier, sizeof(bool) * num_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&visited_ptr, sizeof(bool) * num_nodes);
	cudaMemcpy(visited_ptr, visited, sizeof(bool) * num_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&done, sizeof(bool));

	// Stop timer for preprocessing
	auto end_pp = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_pp_seconds = end_pp - start_pp;
	double duration_pp = duration_pp_seconds.count() * 1000.0;

	// Start timer for BFS traversal
	auto start_bfs = std::chrono::high_resolution_clock::now();

	// Loop until BFS traversal is complete
	do {
		// Initialize the 'done' flag to true
		done_val = true;
		// Copy the 'done' flag from host to device memory
		cudaMemcpy(done, &done_val, sizeof(bool), cudaMemcpyHostToDevice);

		// Launch the BFS kernel to process the current frontier
		BFS_KERNEL << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS >> > (node_ptr, edges_ptr, currqueue, nextqueue, visited_ptr, done, num_nodes_ptr);

		// Copy the 'done' flag from device to host memory
		cudaMemcpy(&done_val, done, sizeof(bool), cudaMemcpyDeviceToHost);
		// Copy the visited array from device to host memory
		cudaMemcpy(visited, visited_ptr, num_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

		// Swap the current and next frontiers
		bool* tmp = currqueue;
		currqueue = nextqueue;
		nextqueue = tmp;
	} while (!done_val);

	// Calculate the count of visited nodes
	for(int i=0; i<num_nodes; i++) count += visited[i];

	// Stop timer for BFS traversal
	auto end_bfs = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_bfs_seconds = end_bfs - start_bfs;
	double duration_bfs = duration_bfs_seconds.count() * 1000.0;

	// Output results
	cout << "Number of nodes in graph: " << num_nodes << endl;
	cout << "Number of edges in graph: " << num_edges << endl;
	cout << "Time taken in preprocessing: " << duration_pp << " ms" << endl;
	cout << "Time taken by BFS: " << duration_bfs << " ms" << endl;
    cout << "Number of nodes visited by BFS: " << count << endl << endl;

	// Free device memory
	cudaFree(node_ptr);
	cudaFree(edges_ptr);
	cudaFree(currqueue);
	cudaFree(nextqueue);
	cudaFree(visited_ptr);
	cudaFree(done);
}

int main()
{
	// Vector to store input graphs
	vector<ECLgraph> store;

	// Read input graphs and store them in the vector
	ECLgraph g1 = readECLgraph("Graphs/Email-Enron.egr");
    ECLgraph g2 = readECLgraph("Graphs/amazon0601.egr");
    ECLgraph g3 = readECLgraph("Graphs/as-skitter.egr");
    ECLgraph g4 = readECLgraph("Graphs/delaunay_n24.egr");
	
	store.push_back(g1);
	store.push_back(g2);
	store.push_back(g3);
	store.push_back(g4);

	// Process each graph in the vector
	for(auto &g: store){
		// Initialize data structures
		memset(curr_frontier, false, sizeof(curr_frontier));
		memset(next_frontier, false, sizeof(next_frontier));
		memset(visited, false, sizeof(visited));
		num_nodes = g.nodes;
		num_edges = g.edges1;
		NUMBER_OF_THREADS = min((1 << 10), num_nodes); // defines the number of threads
		NUMBER_OF_BLOCKS = (num_nodes + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS; // defines the number of blocks

		for (int i = 0; i < num_nodes - 1; i++) {
			node[i].start = g.nindex[i];
			node[i].length = g.nindex[i+1] - g.nindex[i];
		}

		node[num_nodes - 1].start = g.nindex[num_nodes - 1];
		node[num_nodes - 1].length = num_edges - g.nindex[num_nodes - 1];

		for (int i = 0; i < num_edges; i++) edges[i] = g.nlist[i];

		bfs_caller();
	}

    return 0;
}

/*
Sample input graph adjacency list (outgoing edges):
0: 1 2
1: 3 4
2: 3
3: 4
4: 0
*/
/*
Sample input for graph with 5 nodes and 7 edges:
5 7
0 2
2 2
4 1
5 1
6 1
1
2
3
4
3
4
0
*/