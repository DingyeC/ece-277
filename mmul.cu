/*************************************************************************
/* ECE 277: GPU Programmming 2020 
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernal_mmul(int* A, int* W, int* C, const int M, const int N, const int K);

void cu_mmul(int* A, int* W, int* C, const int M, const int N, const int K)
{
	int *d_a, *d_w, *d_c;

	dim3 blk;
	blk.x = 32; blk.y = 32;

    //const int M = 34*34, N = 64, K = 64;
	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int A_size = sizeof(unsigned int)*M*K;
	int W_size = sizeof(unsigned int)*K*N;
	int C_size = sizeof(unsigned int)*M*N;

	cudaMalloc((void **)&d_a, A_size);
	cudaMalloc((void **)&d_w, W_size);
	cudaMalloc((void **)&d_c, C_size);

	cudaMemcpy(d_a, A, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, W, W_size, cudaMemcpyHostToDevice);

	kernal_mmul << < grid, blk >> > (d_a, d_w, d_c, M, N, K);

	cudaMemcpy(C, d_c, C_size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_w);
	cudaFree(d_c);
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
__global__ void kernal_mmul(int* A, int* W, int* C, const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && N < N) {
        int psum = 0;
		#pragma unroll
        for (int k = 0; k < K; k++) {
            psum += A[OFFSET(m, k, K)] * W[OFFSET(k, n, N)];
        }
        C[OFFSET(m, n, N)] = psum;
    }
}

