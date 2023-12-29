#include "opticalFlow_gpu.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <cmath>

using namespace cv;
using namespace std;

opticalFlow_gpu::opticalFlow_gpu(int srcType) : outPoints("gpuOut.yaml")
{
	//opticalFlow_gpu::srcType = srcType;
	
	quailityLevel = 0.3;
	minDistance = 35.0;
	detector = cuda::createGoodFeaturesToTrackDetector(srcType, maxCorners, quailityLevel, minDistance);

	d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create();

	RNG rng;
	for (int i = 0; i < maxCorners; ++i)
	{
		int r = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		colors.push_back(Scalar(r, g, b));
	}
}

void opticalFlow_gpu::download(const cv::cuda::GpuMat & d_mat, std::vector<uchar>& vec, cv::cuda::Stream & stream)
{
	const int num= d_mat.cols;
	vec.resize(num);
	Mat h_mat(1, num, CV_8UC1, (void*)&vec[0]);
	d_mat.download(h_mat, stream);
	//d_mat.row(0).download(Mat(vec), stream);
}

void opticalFlow_gpu::download(const cv::cuda::GpuMat & d_mat, std::vector<Point2f>& vec, cv::cuda::Stream & stream)
{
	const int num = d_mat.cols;
	vec.resize(num);
	Mat h_mat(1, num, CV_32FC2, (void*)&vec[0]);
	d_mat.download(h_mat, stream);
}

void opticalFlow_gpu::upload(cv::cuda::GpuMat & d_mat, cv::Mat& h_mat, cv::cuda::Stream & stream)
{
	d_mat.upload(h_mat, stream);
}

void opticalFlow_gpu::upload(cv::cuda::GpuMat & d_mat, std::vector<cv::Point2f>& vec, cv::cuda::Stream & stream)
{
	if (vec.empty())
	{
		d_mat = cuda::GpuMat(0, 0, d_mat.type());
		return;
	}
	const int num = (int)vec.size();
	Mat h_mat(1, num, CV_32FC2, (void*)&vec[0]);
	d_mat.upload(h_mat, stream);
}

//void opticalFlow_gpu::draw(cv::Mat& frame, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status)
//{
//	betterPts.clear();
//	for (size_t i = 0uLL; i < corners.size(); ++i)
//	{
//		if (1 == status[i])
//		{
//			betterPts.push_back(corners[i]);
//			buffer[i].push_back(corners[i]);
//			
//			circle(frame, nextPts[i], 2, colors[i], -1);
//			if (nextPts[i].y > corners[i].y)
//			{
//				line(frame, corners[i], nextPts[i], colors[i], 2);
//			}
//		}
//		else if (!buffer[i].empty())
//		{
//			pointOut.named() << buffer[i];
//
//			buffer[i].clear();
//		}
//
//	}
//}

void opticalFlow_gpu::sift(Mat& frame, const vector<Point2f>& corners, const vector<Point2f>& backcross, const vector<Point2f>& nextPts, const vector<uchar>& status)
{
	std::vector<std::vector<cv::Point2f> > tracks_new;
//	double dx, dy;
	for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
	{
		if (1 != status[i])
		{
			if (2 < tracks[i].size())
				outPoints.named() << tracks[i];
			continue;
		}
		if (corners[i].y < nextPts[i].y)
		{
			continue;
		}
		double dx = abs(corners[i].x - backcross[i].x);
		double dy = abs(corners[i].y - backcross[i].y);
		if (errorMax < dx + dy)
		{
			//cout << "crossback\n";
			continue;
		}

		//betterPts.push_back(nextPts[i]);
		tracks[i].push_back(nextPts[i]);
		tracks_new.push_back(tracks[i]);
		//Mat Wrapper(1, tracks[i].size(), CV_32FC2, (void*)&tracks[i]);
		//Mat board(1, tracks[i].size(), CV_32SC2);
		//Wrapper.convertTo(board, CV_32SC2);
		//cv::polylines(frame, board, false, colors[i], 2);
		
		if (i < maxCorners)
		{
			//cv::line(frame, corners[i], nextPts[i], colors[i], 2);
			cv::circle(frame, nextPts[i], 3, colors[i], -1);
		}
	}//end sift
	tracks = tracks_new;

	// TODO: 在此处插入 return 语句
}


