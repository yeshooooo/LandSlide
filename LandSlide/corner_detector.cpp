#include "corner_detector.h"

using namespace cv;

corner_detector::corner_detector(int srcType)
{
	//opticalFlow_gpu::srcType = srcType;
	maxCorners = 25;
	quailityLevel = 0.3;
	minDistance = 7.0;
	detector = cuda::createGoodFeaturesToTrackDetector(srcType, maxCorners, quailityLevel, minDistance);

}

inline void corner_detector::detect_corners(cv::cuda::GpuMat & img, cv::cuda::GpuMat & corners, cv::cuda::Stream & stream)
{
	detector->detect(img, corners, noArray(), stream);
}




