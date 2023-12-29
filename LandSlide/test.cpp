#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include "test.h"

using namespace cv;

float dp = 1.f;
float minDist = 10.f;
int cannyThreshold = 40;
int votesThreshold = 20;
int minRadius = 1;
int maxRadius = 255;

cv::Mat h_mat = cv::imread("4.png");

void test()
{
	cuda::GpuMat d_mat(200, 100, CV_8UC3, Scalar(0, 0, 0));
	//circle(d_mat, Point(100, 50), 30, Scalar(255, 100, 100));
	Mat gray;
	cvtColor(h_mat, gray, COLOR_BGRA2GRAY);
	d_mat.upload(gray);
	namedWindow("circle");
	createTrackbar("cannyThreshold\n", "circle", &cannyThreshold, 80, onChange, &d_mat);
	createTrackbar("votesThreshold", "circle", &votesThreshold, 60, onChange, &d_mat);
	waitKey();

	
	//imshow("test", h_mat);
}
void onChange(int param, void * userdata)
{
	cv::cuda::GpuMat d_circles;
	std::vector<cv::Vec3f> circles;
	cuda::GpuMat * src = (cuda::GpuMat*)userdata;
	cv::Ptr<cv::cuda::HoughCirclesDetector> detector = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
	detector->detect(*src, d_circles);
	circles.resize(d_circles.size().width);
	Mat out = Mat::zeros(h_mat.size(), h_mat.type());
	if (!circles.empty())
	{
		d_circles.row(0).download(Mat(circles).reshape(3, 1));
	}
	std::cout << "No" << circles.size() << std::endl;
	
	for (size_t i = 0uLL; i < circles.size(); ++i)
	{
		Vec3i cir = circles[i];
		circle(out, Point(cir[0], cir[1]), cir[2], Scalar(255, 100, 100), 2, LINE_AA);
	}
	add(out, h_mat, out);
	imshow("circle", out);
}