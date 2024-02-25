#include <stdio.h>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int BLOCK_SIZE = 1024;

__global__ void bfs_kernel(int* graph, int* visited, int* queue, int* result, int numNodes, int startNode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int res = 0;
    if (tid == 0) {
        int j;
        int visited[10] = { 0 };
        result[res++] = startNode;
        // printf("%d ", startNode);
        visited[startNode] = 1;

        int queueFront = 0;
        int queueRear = 0;
        queue[queueRear++] = startNode;

        // enqueue(startNode);
        while (queueFront < queueRear) {
            // startNode = dequeue();
            startNode = queue[queueFront++];
            for (j = 0; j < numNodes; j++) {
                if (graph[startNode * numNodes + j] == 1 && visited[j] == 0) {
                    result[res++] = j;
                    // printf("%d ", j);
                    visited[j] = 1;
                    // enqueue(j);
                    queue[queueRear++] = j;
                }
            }
        }
    }
}

int main() {
    auto start_pp = std::chrono::high_resolution_clock::now();
    const int numNodes = 5;
    const int graphSize = numNodes * numNodes * sizeof(int);
    const int resultSize = numNodes * sizeof(int);

    // Initialize graph, visited, and other arrays
    int h_graph[numNodes][numNodes];
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            h_graph[i][j] = rand() % 2; // 0 or 1
        }
    }
    int h_visited[numNodes] = { 0 };
    int h_queue[numNodes];
    int h_result[numNodes];

    int* d_graph, * d_visited, * d_queue, * d_result;

    cudaMalloc((void**)&d_graph, graphSize);
    cudaMalloc((void**)&d_visited, resultSize);
    cudaMalloc((void**)&d_queue, resultSize);
    cudaMalloc((void**)&d_result, resultSize);

    cudaMemcpy(d_graph, h_graph, graphSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, resultSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue, h_queue, resultSize, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto end_pp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pp_seconds = end_pp - start_pp;
    double duration_pp = duration_pp_seconds.count() * 1000.0;

    auto start = std::chrono::high_resolution_clock::now();

    bfs_kernel << <gridDim, blockDim >> > (d_graph, d_visited, d_queue, d_result, numNodes, 0);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seconds = stop - start;
    double duration = duration_seconds.count() * 1000.0;

    std::cout << "Number of nodes in graph: " << numNodes << std::endl;
    std::cout << "Time taken in preprocessing: " << duration_pp << " ms" << std::endl;
    std::cout << "Time taken by BFS: " << duration << " ms" << std::endl;

    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Print or use the result array
    for (int i = 0; i < sizeof(h_result) / sizeof(int); i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");

    cudaFree(d_graph);
    cudaFree(d_visited);
    cudaFree(d_queue);
    cudaFree(d_result);

    return 0;
}
