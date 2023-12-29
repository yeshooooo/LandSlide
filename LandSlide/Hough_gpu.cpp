#include "Hough_gpu.h"
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

Hough_gpu::Hough_gpu(int max)
{
	dp = 1.f;
	minDist = 10.f;
	cannyThreshold = 40;
	votesThreshold = 20;
	minRadius = 1;
	maxRadius = max;
	detector = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
}

void Hough_gpu::run(cv::cuda::GpuMat & src, cv::cuda::Stream & stream)
{
	detector->detect(src, d_circles, stream);
	std::cout << d_circles.size().width << '\t';
	//circles.resize(d_circles.size().width);
	//if (!circles.empty())
	//{
	//	std::cout << "yes";
	//}

	d_circles.release();
	circles.clear();
}

Hough_gpu::~Hough_gpu()
{
}

