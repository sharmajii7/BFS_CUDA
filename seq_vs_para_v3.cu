#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
// #include <device_functions.h>

using namespace std;

// Function for sequential breadth-first search
void sequentialBFS(int start_node, int vertices, int* adjacency_list, int* neighbors)
{
    // Initialize visited nodes and queues for BFS
    vector<bool> visited(vertices, false);
    queue<int> q;

    // Start BFS from the start_node
    q.push(start_node);
    visited[start_node] = true;

    // Process nodes until the queue is empty
    while (!q.empty())
    {
        int front_node = q.front();
        q.pop();

        // Explore all neighbors of the front node
        for (int j = neighbors[front_node]; j < neighbors[front_node + 1]; j++)
        {
            int neighboring_node = adjacency_list[j];
            // Check if the neighbor has been visited
            if (!visited[neighboring_node])
            {
                q.push(neighboring_node);
                visited[neighboring_node] = true;
            }
        }
    }
}

// Kernel function for parallel BFS using CUDA
__global__ void parallelBFS(int start_node, int vertices, int* adjacency_list, int* neighbors, int* currentQueue, int* nextQueue, int* visited)
{
    // Calculate thread ID
    int th_id = threadIdx.x + blockIdx.x * blockDim.x;

    // Declare shared memory for managing queues
    extern __shared__ int queues[];

    // Pointers to manage current and next queue tails in shared memory
    int* currentQueueTail = &queues[0];
    int* nextQueueTail = currentQueueTail + 1;

    /* 
    Initialization:
    The first thread initializes the BFS process by adding the start node to the current queue,
    marking it as visited, and initializing the queue tails.
    */
    if (th_id == 0)
    {
        // Start BFS from the specified start node
        currentQueue[0] = start_node;
        
        // Mark the start node as visited
        visited[start_node] = 1;
        
        // Initialize current and next queue tails
        *currentQueueTail = 1;
        *nextQueueTail = 0;
    }
    __syncthreads();

    /* 
    BFS Loop:
    Each thread explores the neighbors of nodes in the current queue, marking them as visited
    and adding unvisited neighbors to the next queue.
    */
    while (*currentQueueTail > 0)
    {
        // Each thread processes a subset of nodes
        for (int i = th_id; i < *currentQueueTail; i += blockDim.x)
        {
            // Get the node to process
            int front_node = currentQueue[i];

            // Explore all neighbors of the front node
            for (int j = neighbors[front_node]; j < neighbors[front_node + 1]; j++)
            {
                // Get the neighboring node
                int neighboring_node = adjacency_list[j];

                // Check if the neighbor has been visited
                if (visited[neighboring_node] == 0)
                {
                    // Mark the neighbor as visited
                    visited[neighboring_node] = 1;
                    
                    // Add the neighbor to the next queue
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

        // Swap current and next queue tails
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

int main()
{
    // Simulation parameters
    // vector<int> total_nodes = { 15, 50, 600, 6000, 10000 };
    // vector<double> probability = { 0.01, 0.2, 0.4, 0.6 };

    vector<int> total_nodes = { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
    vector<double> probability = {0.6 };

    // Iterate over different probabilities and total nodes
    for (double pro : probability)
    {
        cout << "PROBABILITY: " << pro << endl; 
        vector<int> visited_count;

        for (int vertices : total_nodes)
        {
            // Create the graph
            vector<int> adjacency_list;
            int* neighbors_h = new int[vertices + 1];

            // Initialize adjacency list and neighbors
            /*
            Basically, we iterate through each vertex
            At each vertex, we now again go through all other vertices 
            and randomly put edges with a probability of "pro"
            */
            // So, there is "pro" probability that an edge is there between any two vertices
            for (int i = 0; i < vertices; i++)
            {
                neighbors_h[i] = adjacency_list.size(); 
                // Basically, neighbors_h[i+1] - neighbors_h[i] will give the number of neighbors of vertex i
                for (int j = 0; j < vertices; j++)
                {
                    // rand(): returns random integer value
                    // RAND_MAX: max value that can be returned by the rand() function
                    if (((float)rand() / RAND_MAX < pro) && (i != j))
                        adjacency_list.push_back(j);
                }
            }
            neighbors_h[vertices] = adjacency_list.size();

            // Convert adjacency list from vector to array as Cuda doesn't work with vector
            int* adjacency_list_h = new int[adjacency_list.size()];

            for (int i = 0; i < adjacency_list.size(); i++)
                adjacency_list_h[i] = adjacency_list[i];

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
            int NUMBER_OF_THREADS = min((1 << 10), vertices); // defines the number of threads
	        int NUMBER_OF_BLOCKS = (vertices + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS; // defines the number of blocks

            // Timing variables
            clock_t start, end;
            double time_taken_cpu = 0;
            double time_taken_gpu = 0;
            double t_cpu, t_gpu;
            int SAMPLES = 10;
            int totcount = 0;

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
                cudaMemset(visited_d, 0, vertices * sizeof(int)); // marking all vertices as unvisited initially
                start = clock();

                parallelBFS << <NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>> > (start_node, vertices, adjacency_list_d, neighbors_d, currentQueue_d, nextQueue_d, visited_d);
                
                cudaDeviceSynchronize();
                end = clock();
                t_gpu = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_gpu += t_gpu;

                // Copy results back to host
                cudaMemcpy(h_visited, visited_d, vertices * sizeof(int), cudaMemcpyDeviceToHost);
                int count = 0;
                for(int i=0; i<vertices; i++) count += h_visited[i];

                totcount += count;
            }

            // Calculate and print average times
            time_taken_cpu /= SAMPLES;
            time_taken_gpu /= SAMPLES;

            cout << "Vertices: " << vertices << "\t\t";
            cout << "Sequential: " << fixed << setprecision(2) << time_taken_cpu << " ms" << "\t\t";
            cout << "Parallel: " << fixed << setprecision(2) << time_taken_gpu << " ms" << endl; 
            visited_count.push_back(totcount/SAMPLES);

            // Free host memory
            free(adjacency_list_h);
            free(neighbors_h);
            free(h_visited);
            // Free GPU memory
            cudaFree(adjacency_list_d);
            cudaFree(neighbors_d);
            cudaFree(currentQueue_d);
            cudaFree(nextQueue_d);
            cudaFree(visited_d);
        }
        cout << "No of nodes visited by parallel BFS: ";
        for(int i=0; i<visited_count.size(); i++){ 
            if(i != visited_count.size() - 1) cout << visited_count[i] << ", ";
            else cout << visited_count[i] << endl;
        }
        cout << endl;
    }
    return 0;
}