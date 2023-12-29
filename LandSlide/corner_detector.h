#pragma once
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>


class corner_detector
{
	int maxCorners;
	double quailityLevel;
	double minDistance;
	cv::Ptr<cv::cuda::CornersDetector> detector;
	
public:
	corner_detector(int srcType);
	inline void detect_corners(cv::cuda::GpuMat &img, cv::cuda::GpuMat &corners, cv::cuda::Stream &stream);
};