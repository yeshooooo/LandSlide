#pragma once
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

class morphology_gpu
{
	enum { elementShape = cv::MORPH_ELLIPSE };
	cv::Mat element;
	cv::Ptr<cv::cuda::Filter> openFilter;

public:
	//morphology_gpu();
	morphology_gpu(int srcType);
	//~morphology_gpu();
	inline void run(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
	{
		openFilter->apply(img, img, stream);
	};
};







