#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>

class Hough_gpu
{
	float dp;
	float minDist;
	int cannyThreshold;
	int votesThreshold;
	int minRadius;
	int maxRadius;

	cv::Ptr<cv::cuda::HoughCirclesDetector> detector;
	std::vector<cv::Vec3f> circles;
	cv::cuda::GpuMat d_circles;
public:
	Hough_gpu(int max);
	void run(cv::cuda::GpuMat &src, cv::cuda::Stream &stream);
	~Hough_gpu();
	
};