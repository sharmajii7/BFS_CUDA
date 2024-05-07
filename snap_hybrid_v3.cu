#include <iomanip>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "ECL_GRAPH.h"

#define NUM_NODES 50'000'005
#define NUM_EDGES 300'000'005

using namespace std;

typedef struct
{
	int start;     // Index of first adjacent node in edges_ptr	
	int length;    // Number of adjacent nodes 
} Node;

Node node[NUM_NODES];
int edges[NUM_EDGES];
bool curr_frontier[NUM_NODES] = { false }, next_frontier[NUM_NODES] = { false };
bool visited[NUM_NODES] = { false };

int source = 1;
int num_nodes, num_edges;
int degree[NUM_NODES];
// const int alpha = 14;
// const int beta = 24;
int alpha = 10000;
int beta = 2;

//pointers
int* num_nodes_ptr;
Node* node_ptr; // CSR start and lengths
int* edges_ptr; // CSR edges
bool* currqueue;
bool* nextqueue;
bool* visited_ptr;
bool* done;
unsigned int* nf_ptr;
unsigned int* mf_ptr;
unsigned int* m_unvisited_ptr;
int* degree_ptr;
int* num_edges_ptr;
int* beta_ptr;
int n_visited = 0;

__device__ bool valid_idx(int idx, int* num_nodes_ptr) {
	return idx < *num_nodes_ptr;
}

// Mark the next node in the BFS traversal as true in the next frontier
__device__ void next_true(int idx, bool* nextqueue, unsigned int* nf_ptr, unsigned int* mf_ptr, int* degree_ptr, int* num_nodes_ptr, int* beta_ptr) {
    // Set the next frontier flag for the given index to true
    nextqueue[idx] = true;
    // Atomically increment the number of nodes in the next frontier using maximum value to prevent overflow
    // atomicInc(nf_ptr, INT_MAX);
	int temp = (*num_nodes_ptr) * (*beta_ptr) + (*num_nodes_ptr / 5);
	atomicInc(nf_ptr, temp);
    // Atomically add the degree of the current node to the total frontier size using maximum value to prevent overflow
    atomicAdd(mf_ptr, degree_ptr[idx]);
}

// Mark the current node in the BFS traversal as false in the current frontier
__device__ void current_false(int idx, unsigned int* nf_ptr, unsigned int* mf_ptr, int* degree_ptr, bool* currqueue, int* num_nodes_ptr, int* beta_ptr) {
    // Set the current frontier flag for the given index to false
    currqueue[idx] = false;
    // Atomically decrement the number of nodes in the current frontier using maximum value to prevent overflow
    // atomicDec(nf_ptr, INT_MAX);
	int temp = (*num_nodes_ptr) * (*beta_ptr) + (*num_nodes_ptr / 5);
	atomicDec(nf_ptr, temp);
    // Atomically subtract the degree of the current node from the total frontier size using maximum value to prevent overflow
    atomicSub(mf_ptr, degree_ptr[idx]);
}

// Mark the visited node and update the number of unvisited neighbors
__device__ void visit(int idx, unsigned int* m_unvisited_ptr, int* degree_ptr, bool* visited_ptr) {
    // Mark the current node as visited
    visited_ptr[idx] = true;
    // Atomically subtract the degree of the current node from the total number of unvisited neighbors
	// printf("%u %d\n", *m_unvisited_ptr, degree_ptr[idx]);
    atomicSub(m_unvisited_ptr, degree_ptr[idx]);
	// printf("%u\n", *m_unvisited_ptr);
}

// Kernel function for top-down BFS traversal
__global__ void TOPDOWN_BFS_KERNEL(Node* node_ptr, int* edges_ptr, bool* currqueue, bool* nextqueue, bool* visited_ptr, bool* done, unsigned int* nf_ptr, 
	unsigned int* mf_ptr, unsigned int* m_unvisited_ptr, int* num_nodes_ptr, int* degree_ptr, int* beta_ptr)
{
    // Compute the global thread index
	int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread's id is within the valid range and if the current node is in the current frontier
	if (valid_idx(id, num_nodes_ptr) && currqueue[id])
	{
		// Visit the current node and update the number of unvisited neighbors
		visit(id, m_unvisited_ptr, degree_ptr, visited_ptr);
		// Mark the current node as false in the current frontier and update frontier statistics
		current_false(id, nf_ptr, mf_ptr, degree_ptr, currqueue, num_nodes_ptr, beta_ptr);
		// Get the starting and ending indices of the edges associated with the current node
		int start = node_ptr[id].start;
		int end = start + node_ptr[id].length;

		// Traverse the outgoing edges of the current node
		for (int i = start; i < end; i++)
		{
			// Get the ID of the neighbor node
			int nid = edges_ptr[i];

			// Check if the neighbor node is not in the current frontier and has not been visited
			if (!currqueue[nid] && !visited_ptr[nid])
			{
				// Mark the neighbor node as true in the next frontier and update frontier statistics
				next_true(nid, nextqueue, nf_ptr, mf_ptr, degree_ptr, num_nodes_ptr, beta_ptr);
				// Set the 'done' flag to false to indicate that the BFS is not yet complete
				*done = false;
			}
		}
	}
}

// Kernel function for bottom-up BFS traversal
__global__ void BOTTOMUP_BFS_KERNEL(Node* node_ptr, int* edges_ptr, bool* currqueue, bool* nextqueue, bool* visited_ptr, bool* done, unsigned int* nf_ptr, 
	unsigned int* mf_ptr, unsigned int* m_unvisited_ptr, int* num_nodes_ptr, int* degree_ptr, int* beta_ptr)
{
	// Compute the global thread index
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Check if the thread's id is within the valid range and if the node has not been visited
	if (valid_idx(id, num_nodes_ptr) && !visited_ptr[id])
	{
		// If the current node is in the current frontier
		if (currqueue[id]) {
			// Visit the current node and update the number of unvisited neighbors
			visit(id, m_unvisited_ptr, degree_ptr, visited_ptr);
		}
		else {
			// If the current node is not in the current frontier
			// Get the starting and ending indices of the edges associated with the current node
			int start = node_ptr[id].start;
			int end = start + node_ptr[id].length;

			// Traverse all adjacent vertices (neighbors) of the current node
			for (int i = start; i < end; i++)
			{
				// Get the ID of the neighbor node
				int nid = edges_ptr[i];
				// If the neighbor node is in the current frontier
				if (currqueue[nid])
				{
					// Mark the current node as true in the next frontier and update frontier statistics
					next_true(id, nextqueue, nf_ptr, mf_ptr, degree_ptr, num_nodes_ptr, beta_ptr);
					// Set the 'done' flag to false to indicate that the BFS is not yet complete
					*done = false;
					// Exit the loop after finding one neighbor node in the current frontier
					break;
				}
			}
		}
	}

	// Synchronize all threads in the block before proceeding
	__syncthreads();

	// Update the current frontier flag for the current node
	if (valid_idx(id, num_nodes_ptr))
		current_false(id, nf_ptr, mf_ptr, degree_ptr, currqueue, num_nodes_ptr, beta_ptr);
}

int NUMBER_OF_BLOCKS;
int NUMBER_OF_THREADS;

bool state; // 0 -> TOPDOWN, 1 -> BOTTOMUP
//TODO : account for increase or decrease in nodes and edges in frontier

int n_top_down = 0;
int n_bottom_up = 0;

void bfs_caller()
{
	auto start_pp = std::chrono::high_resolution_clock::now();
	int nf = 1;
    int count = 0;
	state = 0;
	int mf = degree[source];
	int m_unvisited = num_edges;
	bool done_val = true;
	curr_frontier[source] = true;
	/*
	nf:
	- Refers to the number of nodes in the current frontier.
	- It represents the size of the frontier, i.e., 
	  the set of nodes that are currently being explored in the BFS traversal.

	mf:
	- Measure of the frontier.
	- It can represent various metrics related to the frontier, such as the sum of the degrees of nodes in the frontier.
	- In our code, mf represents the accumulated measure of the frontier, such as the sum of degrees of nodes in the current frontier.

	These metrics (nf and mf) are used to dynamically switch between top-down and bottom-up BFS traversal strategies based on certain conditions.


	The transition from top-down to bottom-up BFS might occur 
	when the measure of the frontier (mf) exceeds a certain threshold relative to the 
	number of unvisited nodes (m_unvisited), indicating that exploring the frontier 
	from the bottom-up could be more efficient.

	Conversely, the transition from bottom-up to top-down BFS might occur 
	when the number of nodes in the frontier (nf) falls below a certain threshold relative to the 
	total number of nodes (num_nodes), suggesting that exploring the frontier 
	from the top-down might be more efficient.
	*/

	cudaMalloc((void**)&num_nodes_ptr, sizeof(int));
	cudaMemcpy(num_nodes_ptr, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&nf_ptr, sizeof(unsigned int));
	cudaMemcpy(nf_ptr, &nf, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&mf_ptr, sizeof(unsigned int));
	cudaMemcpy(mf_ptr, &mf, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_unvisited_ptr, sizeof(unsigned int));
	cudaMemcpy(m_unvisited_ptr, &num_edges, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&num_edges_ptr, sizeof(unsigned int));
	cudaMemcpy(num_edges_ptr, &num_edges, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&beta_ptr, sizeof(int));
	cudaMemcpy(beta_ptr, &beta, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&degree_ptr, sizeof(int) * num_nodes);
	cudaMemcpy(degree_ptr, degree, sizeof(int) * num_nodes, cudaMemcpyHostToDevice);


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

	auto end_pp = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_pp_seconds = end_pp - start_pp;
	double duration_pp = duration_pp_seconds.count() * 1000.0;

	auto start_bfs = std::chrono::high_resolution_clock::now();
	do {
		done_val = true;
		cudaMemcpy(done, &done_val, sizeof(bool), cudaMemcpyHostToDevice);

		// cout << "nf: " << nf << "\t\t" << "mf: " << mf << "\t" << endl;
		// cout << "m_unvisited: " << m_unvisited << "\t" << "num_nodes: " << num_nodes << endl;
		
		if (!state && mf > m_unvisited / alpha) {
			printf("Going from top-down to bottom-up\n");
			state = 1;
		}
		else if (state && nf < num_nodes / beta) {
			printf("Going from bottom-up to top-down\n");
			state = 0;
		}

		if (!state) {
			// printf("Going top-down\n");
			n_top_down++;
			TOPDOWN_BFS_KERNEL << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS >> > (node_ptr, edges_ptr, currqueue, nextqueue, visited_ptr, done, nf_ptr, mf_ptr, m_unvisited_ptr, num_nodes_ptr, degree_ptr, beta_ptr);
		}
		else {
			// printf("Going bottom-up\n");
			n_bottom_up++;
			BOTTOMUP_BFS_KERNEL << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS >> > (node_ptr, edges_ptr, currqueue, nextqueue, visited_ptr, done, nf_ptr, mf_ptr, m_unvisited_ptr, num_nodes_ptr, degree_ptr, beta_ptr);
		}

		cudaMemcpy(&done_val, done, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&mf, mf_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&nf, nf_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&m_unvisited, m_unvisited_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		// Copy the visited array from device to host memory
		cudaMemcpy(visited, visited_ptr, num_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

		// Swapping current and next frontiers..
		bool* tmp = currqueue;
		currqueue = nextqueue;
		nextqueue = tmp;
	} while (!done_val);

	// Calculate the count of visited nodes
	for(int i=0; i<num_nodes; i++) count += visited[i];

	auto end_bfs = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_bfs_seconds = end_bfs - start_bfs;
	double duration_bfs = duration_bfs_seconds.count() * 1000.0;

	cout << "Number of nodes in graph: " << num_nodes << endl;
	cout << "Number of edges in graph: " << num_edges << endl;
	cout << "Time taken in preprocessing: " << duration_pp << " ms" << endl;
	cout << "Time taken by Hybrid BFS: " << duration_bfs << " ms" << endl;
    cout << "Number of nodes visited by BFS: " << count << endl << endl;
 
	cudaFree(node_ptr);
	cudaFree(edges_ptr);
	cudaFree(currqueue);
	cudaFree(nextqueue);
	cudaFree(visited_ptr);
	cudaFree(done);
	cudaFree(nf_ptr);
	cudaFree(mf_ptr);
	cudaFree(m_unvisited_ptr);
	cudaFree(num_edges_ptr);
	cudaFree(degree_ptr);
}

// The BFS frontier corresponds to all the nodes being processed at the current level.

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

		alpha = num_nodes;
		alpha /= 1000;
		beta = 2;

		for (int i = 0; i < num_nodes - 1; i++) {
			node[i].start = g.nindex[i];
			node[i].length = g.nindex[i+1] - g.nindex[i];
			degree[i] = g.nindex[i+1] - g.nindex[i];
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