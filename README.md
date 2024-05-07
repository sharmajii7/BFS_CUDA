# BFS_CUDA
Parallelization of Breadth First Search using CUDA 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**snap_para_v2.cu:**

In this file, I have implemented BFS (Breadth First Search) using CUDA but only utilised the top-down kernel,
i.e., there is no switching being done between top-down and bottom-up kernel. 
Code was implemented by making appropriate changes to the OpenMP code explained in this video: https://youtu.be/SKhMrCaaduU?si=25mPKc6vsmO_UtHp . 
I have used SNAP graphs in .egr format {CSR format} to save to make the program memory efficient.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**seq_vs_para_v3.cu:**

In this file, I made two functions, in which one of them does normal sequential BFS and the other one does parallel BFS using CUDA.
Speed comparison is done for different sizes of graphs (namely: 15, 50, 600, 6000, 10000) and different edge probabilities (namely: 0.01, 0.2, 0.4, 0.6). 
This program demonstrates the true power of parallelisation in big graphs.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**snap_hybrid_v3.cu:**

In this file, I have implemented Hybrid BFS, which is just a combination of Top-Down BFS and Bottom-Up BFS. The input format for this file is CSR (Compressed Sparse Row) format for representing graphs. 
This provides major space reduction for graphs of large sizes. It also provides a noticable speed boost as most of the time in parallisation programs is consumed in precomputation (CudaMemcpy, etc.).
