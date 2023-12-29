#include "Header.cuh"
#include <iostream>
#include "device_launch_parameters.h"
//#include "device_functions.h"
#include <cmath>
#include "cuda_runtime_api.h"


using namespace std;
using namespace cv;

__global__ void mergeKernel(cuda::PtrStepSz<float2> out, cuda::PtrStepSz<float2> mat1, cuda::PtrStepSz<float2> mat2)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; //列坐标
	//int j = threadIdx.y + blockIdx.y * blockDim.y; //行坐标

	if (i < mat1.cols) // 必须要检查是否越界
	{
		out(0, i) = mat1(0, i);
	}
	else
	{
		out(0, i) = mat2(0, i - mat1.cols);
	}
	//__syncthreads();
}

__global__ void siftKernel(cuda::PtrStepSz<float2> out, cuda::PtrStepSz<float2> mat1, cuda::PtrStepSz<float2> mat2, cuda::PtrStepSz<uchar> status, int* index)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (1 != status(0, id))
	{
		return;
	}
	if (mat1(0, id).y > mat2(0, id).y)
		return;
	float dx = (mat1(0, id).x - mat2(0, id).x);
	float dy = (mat1(0, id).y - mat2(0, id).y);
	if (3 < dx + dy)
	{
		return;
	}
	out(0, *index) = out(0, id);
	atomicAdd(index, 1);
}

cudaError_t callback(cv::cuda::GpuMat &d_better, cv::cuda::GpuMat &d_nuwP) {
	//int *dev_a = nullptr;
	int col = d_nuwP.size().width;
	if (1 != d_nuwP.rows)
	{
		cerr << "sth wrong\n";
		exit(-1);

	}
	//size_t gridSize = 2;
	dim3 blockSize(1, col / 2);
	//mat.operator cuda::PtrStepSz<uchar1>;
	//cudaStream_t stream 

	//CE(cudaSetDevice(0));

	//CE(cudaMalloc((void**)&dev_a, size * sizeof(int)));

	//CE(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

	//size_t blockSize = size / 2;  // block的大小是size/2
	//kernel << <gridSize, blockSize >> > (mat);

	//CE(cudaGetLastError());
	//CE(cudaDeviceSynchronize());

	// 从显存中把数据复制回内存
	//CE(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	//CE(cudaFree(dev_b));
	//CE(cudaDeviceReset());
	return cudaError_t();
}

cv::cuda::GpuMat merge(cv::cuda::GpuMat &d_better, cv::cuda::GpuMat &d_nuwP)
{
	int col1 = d_better.cols;
	int col2 = d_nuwP.cols;
	int col = (col1 + col2);
	cuda::GpuMat out(1, col, d_better.type());
	mergeKernel << <1, col>> > (out, d_better, d_nuwP);
	CE(cudaGetLastError());
	CE(cudaDeviceSynchronize());
	return out;
}

void sift(cv::cuda::GpuMat &corners, cv::cuda::GpuMat &backcross, cv::cuda::GpuMat &nextPtr, cv::cuda::GpuMat &status)
{
	int col = nextPtr.cols;
	//int *index = new int[nextPtr.cols];
	int * index;
	int * tmp = new int;
	CE(cudaMalloc((void**)&index, sizeof(int)));
	cudaMemset(index, 0, sizeof(int));
	//cudaMallocManaged(&index, 0);
	//CE(cudaMemset((void**)&index, sizeof(int)));
	siftKernel << <1, col >> > (nextPtr, corners, backcross, status, index);
	CE(cudaMemcpy(tmp, index, sizeof(int), cudaMemcpyDeviceToHost));
	nextPtr = nextPtr.colRange(0, *tmp);
	CE(cudaGetLastError());
	CE(cudaDeviceSynchronize());
	CE(cudaFree(index));
	delete tmp;
}

