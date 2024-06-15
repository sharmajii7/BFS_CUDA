// CUDA C/C++ implementation for Accelerating Graph Betweenness Centrality for Sparse Graphs

#include <iostream>
#include <cuda.h>
#include "Graph.h"

#define MAX_THREAD_COUNT 1024
#define CEIL(a, b) ((a - 1) / b + 1)

using namespace std;

#define catchCudaError(error) { gpuAssert((error), __FILE__, __LINE__); }

float device_time_taken;

// Catch Cuda errors
inline void gpuAssert(cudaError_t error, const char *file, int line,  bool abort = false)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s in CUDA %s\n", error, cudaGetErrorString(error));
        printf("\nIn file :%s\nOn line: %d", file, line);
        
        if(abort)
            exit(-1);
    }
}

__global__ void betweennessCentralityKernel(Graph *graph, double *bwCentrality, int nodeCount,
            int *sigma, int *distance, double *dependency, int *Q, int *Qpointers) {
    
    int idx = threadIdx.x;
    if(idx >= nodeCount)
        return;
    
    __shared__ int s;
    __shared__ int Q_len;
    __shared__ int Qpointers_len;

    if(idx == 0) {
        s = -1;
        // printf("Progress... %3d%%", 0);
    }
    __syncthreads();

    while(s < nodeCount -1)
    {    
        if(idx == 0)
        {
            ++s;
            // printf("\rProgress... %5.2f%%", (s+1)*100.0/nodeCount);
            
            Q[0] = s;
            Q_len = 1;
            Qpointers[0] = 0;
            Qpointers[1] = 1;
            Qpointers_len = 1;
        }
        __syncthreads();

        for(int v=idx; v<nodeCount; v+=blockDim.x)
        {
            if(v == s)
            {
                distance[v] = 0;
                sigma[v] = 1;
            }
            else
            {
                distance[v] = INT_MAX;
                sigma[v] = 0;
            }
            dependency[v] = 0.0;
        }
        __syncthreads();
        
        // BFS
        while(true)
        {
            __syncthreads();
            for(int k=idx; k<Qpointers[Qpointers_len]; k+=blockDim.x)
            {
                if(k < Qpointers[Qpointers_len -1])
                    continue;

                int v = Q[k];
                for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                {
                    int w = graph->adjacencyList[r];
                    if(atomicCAS(&distance[w], INT_MAX, distance[v] +1) == INT_MAX)
                    {
                        int t = atomicAdd(&Q_len, 1);
                        Q[t] = w;
                    }
                    if(distance[w] == (distance[v]+1))
                    {
                        atomicAdd(&sigma[w], sigma[v]);
                    }
                }
            }
            __syncthreads();

            if(Q_len == Qpointers[Qpointers_len])
                break;

            if(idx == 0)
            {
                Qpointers_len++;
                Qpointers[Qpointers_len] = Q_len;
            }
            __syncthreads();
        }
        __syncthreads();
        
        // Reverse BFS
        while(Qpointers_len > 0)
        {
            for(int k=idx; k < Qpointers[Qpointers_len]; k+=blockDim.x) 
            {
                if(k < Qpointers[Qpointers_len -1])
                    continue;

                int v = Q[k];
                for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                {
                    int w = graph->adjacencyList[r];
                    if(distance[w] == (distance[v] + 1))
                    {
                        if (sigma[w] != 0)
                            dependency[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + dependency[w]);
                    }
                }
                if (v != s)
                {
                    // Each shortest path is counted twice. So, each partial shortest path dependency is halved.
                    bwCentrality[v] += dependency[v] / 2;
                }
            }
            __syncthreads();

            if(idx == 0)
                Qpointers_len--;

            __syncthreads();
        }
    }
}

double *betweennessCentrality(Graph *graph, int nodeCount)
{
    double *bwCentrality = new double[nodeCount]();
    double *device_bwCentrality, *dependency;
    int *sigma, *distance, *Q, *Qpointers;

    catchCudaError(cudaMalloc((void **)&device_bwCentrality, sizeof(double) * nodeCount));
    catchCudaError(cudaMalloc((void **)&sigma, sizeof(int) * nodeCount));
    catchCudaError(cudaMalloc((void **)&distance, sizeof(int) * nodeCount));
    catchCudaError(cudaMalloc((void **)&Q, sizeof(int) * (nodeCount +1)));
    catchCudaError(cudaMalloc((void **)&Qpointers, sizeof(int) * (nodeCount +1)));
    catchCudaError(cudaMalloc((void **)&dependency, sizeof(double) * nodeCount));
    catchCudaError(cudaMemcpy(device_bwCentrality, bwCentrality, sizeof(double) * nodeCount, cudaMemcpyHostToDevice));

    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start));
    catchCudaError(cudaEventCreate(&device_end));
    catchCudaError(cudaEventRecord(device_start));

    betweennessCentralityKernel<<<1, MAX_THREAD_COUNT>>>(graph, device_bwCentrality, nodeCount, sigma, distance, dependency, Q, Qpointers);
    cudaDeviceSynchronize();

    // Timer
    catchCudaError(cudaEventRecord(device_end));
    catchCudaError(cudaEventSynchronize(device_end));
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    // Copy back and free memory
    catchCudaError(cudaMemcpy(bwCentrality, device_bwCentrality, sizeof(double) * nodeCount, cudaMemcpyDeviceToHost));
    catchCudaError(cudaFree(device_bwCentrality));
    catchCudaError(cudaFree(sigma));
    catchCudaError(cudaFree(dependency));
    catchCudaError(cudaFree(distance));
    catchCudaError(cudaFree(Q));
    catchCudaError(cudaFree(Qpointers));
    return bwCentrality;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << "Please use correct format while execution" << endl;
        return 0;
    }

    // char choice;
    // cout << "Would you like to print the Graph Betweenness Centrality for all nodes? (y/n) ";
    // cin >> choice;

    freopen(argv[1], "r", stdin);

    Graph *host_graph = new Graph();
    Graph *device_graph;

    catchCudaError(cudaMalloc((void **)&device_graph, sizeof(Graph)));
    host_graph->readGraph();

    int nodeCount = host_graph->getNodeCount();
    int edgeCount = host_graph->getEdgeCount();
    catchCudaError(cudaMemcpy(device_graph, host_graph, sizeof(Graph), cudaMemcpyHostToDevice));

    // Copy Adjancency List to device
    int *adjacencyList;
    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&adjacencyList, sizeof(int) * (2 * edgeCount + 1)));
    catchCudaError(cudaMemcpy(adjacencyList, host_graph->adjacencyList, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice));
    // Update the pointer to this, in device_graph
    catchCudaError(cudaMemcpy(&(device_graph->adjacencyList), &adjacencyList, sizeof(int *), cudaMemcpyHostToDevice));

    // Copy Adjancency List Pointers to device
    int *adjacencyListPointers;
    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&adjacencyListPointers, sizeof(int) * (nodeCount + 1)));
    catchCudaError(cudaMemcpy(adjacencyListPointers, host_graph->adjacencyListPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice));
    // Update the pointer to this, in device_graph
    catchCudaError(cudaMemcpy(&(device_graph->adjacencyListPointers), &adjacencyListPointers, sizeof(int *), cudaMemcpyHostToDevice));

    double *bwCentrality = betweennessCentrality(device_graph, nodeCount);

    double maxBetweenness = -1;
    for (int i = 0; i < nodeCount; i++)
    {
        maxBetweenness = max(maxBetweenness, bwCentrality[i]);
        // if (choice == 'y' || choice == 'Y')
        //     printf("Node %d => Betweeness Centrality %0.2lf\n", i, bwCentrality[i]);
    }

    // cout << endl;

    printf("Maximum Betweenness Centrality = %0.2lf\n", maxBetweenness);
    printf("Time Taken (Parallel) = %d ms\n", (int)device_time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << bwCentrality[i] << " ";
        cout << endl;
    }

    // Free all memory
    catchCudaError(cudaFree(adjacencyList));
    catchCudaError(cudaFree(adjacencyListPointers));
    catchCudaError(cudaFree(device_graph));
}