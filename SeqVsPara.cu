#include <iostream>
#include <vector>
#include <queue>
#include <ctime>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "device_functions.h"

using namespace std;

// Function for sequential breadth-first search
void sequentialBFS(int start_node, int vertices, int* adjacency_list, int* neighbors)
{
    // Initialize visited nodes and queues for BFS
    vector<bool> visited(vertices, false);
    queue<int> temprorary_queue;
    queue<int> output_queue;

    // Start BFS from the start_node
    temprorary_queue.push(start_node);
    output_queue.push(start_node);
    visited[start_node] = true;

    // Process nodes until the queue is empty
    while (!temprorary_queue.empty())
    {
        int front_node = temprorary_queue.front();
        temprorary_queue.pop();

        // Explore all neighbors of the front node
        for (int j = neighbors[front_node]; j < neighbors[front_node + 1]; j++)
        {
            int neighboring_node = adjacency_list[j];
            // Check if the neighbor has been visited
            if (!visited[neighboring_node])
            {
                output_queue.push(neighboring_node);
                temprorary_queue.push(neighboring_node);
                visited[neighboring_node] = true;
            }
        }
    }
}

// Kernel function for parallel BFS using CUDA
__global__ void parallelBFS(int start_node, int vertices, int* adjacency_list, int* neighbors, int* currentQueue, int* nextQueue, int* visited)
{
    int th_id = threadIdx.x * blockDim.x * threadIdx.x;
    extern __shared__ int queues[];

    // Shared memory for managing queues
    int* currentQueueTail = &queues[0];
    int* nextQueueTail = currentQueueTail + 1;

    // Initialization by the first thread
    if (th_id == 0)
    {
        currentQueue[0] = start_node;
        visited[start_node] = 1;
        *currentQueueTail = 1;
        *nextQueueTail = 0;
    }
    __syncthreads();

    // Loop while there are nodes to process
    while (*currentQueueTail > 0)
    {
        // Each thread processes a subset of nodes
        for (int i = th_id; i < *currentQueueTail; i += blockDim.x)
        {
            int front_node = currentQueue[i];

            // Explore all neighbors of the front node
            for (int j = neighbors[front_node]; j < neighbors[front_node + 1]; j++)
            {
                int neighboring_node = adjacency_list[j];

                // Check if the neighbor has been visited
                if (visited[neighboring_node] == 0)
                {
                    visited[neighboring_node] = 1;
                    int pos = atomicAdd(nextQueueTail, 1);
                    nextQueue[pos] = neighboring_node;
                }
            }
        }
        __syncthreads();
        // Swap current and next queues
        int* temp = currentQueue;
        currentQueue = nextQueue;
        nextQueue = temp;

        temp = currentQueueTail;
        currentQueueTail = nextQueueTail;
        nextQueueTail = temp;

        // Reset the next queue tail for the next iteration
        if (th_id == 0)
        {
            *nextQueueTail = 0;
        }
        __syncthreads();
    }
}

int main(void)
{
    // Simulation parameters
    vector<int> total_nodes = { 15, 50, 600, 6000, 10000 };
    vector<double> probability = { 0.01, 0.2, 0.4, 0.6 };

    // Iterate over different probabilities and total nodes
    for (double pro : probability)
    {
        printf("PROBABILITY: %f\n", pro);
        for (int vertices : total_nodes)
        {
            // Create the graph
            vector<int> adjacency_list;
            int* neighbors_h = new int[vertices + 1];

            // Initialize adjacency list and neighbors
            for (int i = 0; i < vertices; i++)
            {
                neighbors_h[i] = adjacency_list.size();
                for (int j = 0; j < vertices; j++)
                {
                    if ((float)rand() / RAND_MAX < pro)
                    {
                        if (i != j)
                        {
                            adjacency_list.push_back(j);
                        }
                    }
                }
            }
            neighbors_h[vertices] = adjacency_list.size();

            // Convert adjacency list to array
            int* adjacency_list_h = new int[adjacency_list.size()];
            for (int i = 0; i < adjacency_list.size(); i++)
            {
                adjacency_list_h[i] = adjacency_list[i];
            }

            // Allocate memory for visited nodes
            int* h_visited = new int[vertices];

            // GPU memory allocations
            int* adjacency_list_d, * neighbors_d, * currentQueue_d, * nextQueue_d, * visited_d;
            cudaMalloc((void**)&adjacency_list_d, adjacency_list.size() * sizeof(int));
            cudaMalloc((void**)&neighbors_d, (vertices + 1) * sizeof(int));
            cudaMalloc((void**)&currentQueue_d, vertices * sizeof(int));
            cudaMalloc((void**)&nextQueue_d, vertices * sizeof(int));
            cudaMalloc((void**)&visited_d, vertices * sizeof(int));

            // Copy data from host to device
            cudaMemcpy(adjacency_list_d, adjacency_list_h, adjacency_list.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(neighbors_d, neighbors_h, (vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

            // Kernel execution parameters
            int NUMBER_OF_BLOCKS = 526;
            int NUMBER_OF_THREADS = 128;
            int SHARED_MEMORY_SIZE = 2 * sizeof(int);

            // Timing variables
            clock_t start, end;
            double time_taken_cpu = 0;
            double time_taken_gpu = 0;
            double t_cpu, t_gpu;
            int SAMPLES = 10;

            // Perform BFS for several start nodes
            for (int start_node = 0; start_node < SAMPLES; start_node++)
            {
                // Sequential BFS
                start = clock();
                sequentialBFS(start_node, vertices, adjacency_list_h, neighbors_h);
                end = clock();
                t_cpu = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_cpu += t_cpu;

                // Parallel BFS
                cudaMemset(visited_d, 0, vertices * sizeof(int));
                start = clock();
                parallelBFS << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >> > (start_node, vertices, adjacency_list_d, neighbors_d, currentQueue_d, nextQueue_d, visited_d);
                cudaDeviceSynchronize();
                end = clock();
                t_gpu = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_gpu += t_gpu;

                // Copy results back to host
                cudaMemcpy(h_visited, visited_d, vertices * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // Calculate and print average times
            time_taken_cpu /= SAMPLES;
            time_taken_gpu /= SAMPLES;
            printf("Vertices: %d\t\tSequential: %.4f ms\t\tParallel: %.4f ms\n", vertices, time_taken_cpu, time_taken_gpu);

            // Free host memory
            free(adjacency_list_h);
            free(neighbors_h);
            free(h_visited);          // Free GPU memory
            cudaFree(adjacency_list_d);
            cudaFree(neighbors_d);
            cudaFree(currentQueue_d);
            cudaFree(nextQueue_d);
            cudaFree(visited_d);
        }
        printf("\n\n");
    }

    return 0;
}
