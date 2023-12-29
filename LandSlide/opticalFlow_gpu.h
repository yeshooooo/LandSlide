#pragma once
#include "out.h"
#include <opencv2/core.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>
#include <array>

class opticalFlow_gpu
{
	out outPoints;
	cv::Size winSize;
	int maxLevel;
	int iters;

	const int errorMax = 4;

	static const int maxCorners = 15;
	double quailityLevel;
	double minDistance;
	
public:
	explicit opticalFlow_gpu(int srcType);
	inline void detect_corners(cv::cuda::GpuMat &img, cv::cuda::GpuMat &corners, cv::InputArray &mask, cv::cuda::Stream &stream)
	{
		detector->detect(img, corners, mask, stream);
	};
	static void download(const cv::cuda::GpuMat &d_mat, std::vector<uchar> &vec, cv::cuda::Stream &stream);
	static void download(const cv::cuda::GpuMat &d_mat, std::vector<cv::Point2f> &vec, cv::cuda::Stream &stream);
	static void upload(cv::cuda::GpuMat & d_mat, cv::Mat& mat, cv::cuda::Stream & stream);
	static void upload(cv::cuda::GpuMat &d_mat, std::vector<cv::Point2f> &vec, cv::cuda::Stream &stream);
	//void draw(cv::Mat& frame, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status);
	void sift(cv::Mat& frame, const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p0r, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status);
	//void draw(cv::Mat& frame, const std::vector<cv::Point2f>& corners);

	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse;
	cv::Ptr<cv::cuda::CornersDetector> detector;

	std::vector<cv::Scalar> colors;
	//std::vector<cv::Point2f> betterPts;

	std::vector<std::vector<cv::Point2f> > tracks;
	
	//int num;
};
