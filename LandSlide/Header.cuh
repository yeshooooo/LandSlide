#ifndef header_cuh_
#define header_cuh_
#include <cstdlib>
#include "cuda_runtime.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

//__global__ void kernel(cv::cuda::PtrStepSz<cv::Point2f> mat);

cv::cuda::GpuMat merge(cv::cuda::GpuMat &d_better, cv::cuda::GpuMat &d_nuwP);
void sift(cv::cuda::GpuMat &corners, cv::cuda::GpuMat &backcross, cv::cuda::GpuMat &nextPtr, cv::cuda::GpuMat &status);

#define CE(call)															\
do																			 \
{																			\
	const cudaError_t error_code = call;									\
	if (cudaSuccess != error_code)											\
	{																		\
		printf("CUDA Error:\n");											\
		printf("File: %s\n", __FILE__);										\
		printf("Line: %d\n", __LINE__);										\
		printf("Error code: %d\n", error_code);								\
		printf("Error text: %s\n", cudaGetErrorString(error_code) );		\
		exit(EXIT_FAILURE);															\
	}																		\
}while(0)

//#if __CUDA_ARCH__ < 620
//__device__ double atomicAdd(double* address, double val)
//{
//	unsigned long long int* address_as_ull =
//		(unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,
//			__double_as_longlong(val +
//				__longlong_as_double(assumed)));
//
//		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//	} while (assumed != old);
//
//	return __longlong_as_double(old);
//}
//#endif

#endif // !header_cuh_


