#include "gpu.h"
#include "morphology_gpu.h"
//#include "Hough_gpu.h"
//#include "corner_detector.h"
#include "opticalFlow_gpu.h"
#include "Header.cuh"
#include <iostream>
#include <string>
#include <numeric>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/opengl.hpp>


#include <opencv2/cudabgsegm.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;

void gpu()
{
	namedWindow("cpu");
	//namedWindow("gpu", WINDOW_OPENGL | WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);//| WINDOW_GUI_EXPANDED);
	//namedWindow("fgmask", WINDOW_OPENGL | WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);
	//cuda::setGlDevice();

	cuda::Stream stream2, s3, s4, s5;

	cuda::GpuMat d_frame, d_fgmask, d_corners, d_gray, d_new, d_mask;
	cuda::GpuMat d_pre, d_nextPtr, d_status, d_err, d_backcross;
	Mat mask, frame, fgMask;
	
	//ogl::Texture2D out;
	Ptr<cudacodec::VideoReader> d_reader = cudacodec::createVideoReader(videoName);
	//VideoCapture cap(videoName);
	//d_reader->set(cudacodec::ColorFormat::BGR);
	Ptr<cuda::BackgroundSubtractorMOG2> mog2 = cuda::createBackgroundSubtractorMOG2(100, 16.0, false);
	d_reader->nextFrame(d_frame, stream2);
	//cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY, 0, stream2);
	
	mog2->apply(d_frame, d_pre);
	d_pre.download(fgMask);
	d_frame.download(frame);
	Mat board(frame.size(), frame.type(), Scalar(0,0,0));
	Mat init(fgMask.size(), fgMask.type(), Scalar(255));
	
	//d_pre.copyTo(d_fgmask, stream2);
	//const int max = cv::max(d_frame.size().width, d_frame.size().width);
	const int type = d_pre.type();
	morphology_gpu morphologyOpen(type);
	//Hough_gpu circlesDetector(max);
	//corner_detector cornersDetector(type);
	opticalFlow_gpu flow(type);
	flow.detect_corners(d_pre, d_corners, noArray(), stream2);


	TickMeter tm;
	std::vector<double> gpu_times(1900);
	//int gpu_frame_count = 0;
	
	int keyboard;
	int frameIndex = 0;
	std::vector<cv::Point2f> corners, backcross;

	std::vector<cv::Point2f> nextPts, newP;
	std::vector<uchar> status;

	//tm.start();
	

	//
	//while (cap.read(frame))
	while (d_reader->nextFrame(d_frame, stream2))
	{
		tm.reset();  
		tm.start();
		//long long start = getTickCount();

		//cout << d_frame.channels() << endl;
		//cout << d_frame.data << endl;
		//cout << d_frame.dataend << endl;
		//cout << d_frame.flags << endl;
		//d_frame.upload(frame, stream2);
		//d_pre.download(frame, stream2);
		//cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY, 0, stream2);
		//cuda::equalizeHist(d_img, d_img, stream2);
		//s3.waitForCompletion();
		mog2->apply(d_frame, d_fgmask, -1.0, stream2);
		
		//mog2->getBackgroundImage(d_bgimg);
		//cuda::threshold(d_fgmask, d_fgmask, 200., 255., THRESH_BINARY, stream2);
		morphologyOpen.run(d_fgmask, stream2);
		//d_fgmask.download(fgMask, stream2);
		//circlesDetector.run(d_fgmask, stream2);
		d_frame.download(frame, s5);
		//if (0 < flow.tracks.size())
		if (!d_corners.empty())
		{
			flow.d_pyrLK_sparse->calc(d_pre, d_fgmask, d_corners, d_nextPtr, d_status, d_err, stream2);

			opticalFlow_gpu::download(d_corners, corners, s3);
			
			//sift(d_corners, d_backcross, d_nextPtr, d_status);
			opticalFlow_gpu::download(d_nextPtr, nextPts, s4);
			opticalFlow_gpu::download(d_status, status, s5);
			flow.d_pyrLK_sparse->calc(d_fgmask, d_pre, d_nextPtr, d_backcross, d_status, cv::noArray(), stream2);
			opticalFlow_gpu::download(d_backcross, backcross, stream2);
			stream2.waitForCompletion();
			//s4.waitForCompletion();
			//flow.sift(board, corners, backcross, nextPts, status);
			flow.sift(board, corners, backcross, nextPts, status);
			
		}
		
		if (0 == frameIndex % 5)
		{
			init.copyTo(mask);
			if (0 < nextPts.size())//过滤已有点周围区域
			{
				for (int i = 0; i < flow.tracks.size(); ++i)
				{
					cv::circle(mask, nextPts[i], 5, Scalar(0), -1);
				}
			}
			opticalFlow_gpu::upload(d_mask, mask, s3);
			flow.detect_corners(d_fgmask, d_new, d_mask ,s3);
			if (0 < d_new.size().width)
			{
				opticalFlow_gpu::download(d_new, newP, s3);
				s3.waitForCompletion();
				for (Point2f x : newP)
				{
					vector<Point2f> tmp{ x };
					flow.tracks.push_back(tmp);
				}
			}
			
			//if (0 < d_new.size().width)
			//	d_corners = merge(d_nextPtr, d_new);
			//opticalFlow_gpu::download(d_corners, corners, stream2);
			//for (int i = 0; i < newP.size(); ++i)
			//{
			//	flow.betterPts.push_back(newP[i]);
			//}
		}
		if (0 == frameIndex % 250)
		{
			board = Mat::zeros(frame.size(), frame.type());
		}
		//opticalFlow_gpu::upload(d_corners, flow.betterPts, stream2);
		//out.copyFrom(d_frame);
		//out.copyFrom(d_img);
		//out.copyFrom(d_fgmask);
		//imshow("gpu", out);
		//d_nextPtr.copyTo()
		d_fgmask.copyTo(d_pre, s4);
		corners.clear();
		for (int i = 0; i < flow.tracks.size(); ++i)//init corners
		{
			corners.push_back(flow.tracks[i].back());
		}
		opticalFlow_gpu::upload(d_corners, corners, stream2);

		//String trackNum = "Tracking: " + to_string(flow.tracks.size());
		//cv::putText(frame, trackNum.c_str(), Point(15, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		add(board, frame, frame);
		//s3.waitForCompletion();
		imshow("cpu", frame);
		++frameIndex;

		//double fps = getTickFrequency() / (getTickCount() - start);
		//cout << "FPS" << fps << "\t帧生成时间：" << 1000. / fps << "ms" << endl;
		tm.stop();
		gpu_times.push_back(tm.getTimeMilli());
		//std::cout << "用时：" << tm.getTimeMilli << std::endl;
		keyboard = waitKey(5);
		if (0 < keyboard)
			break;
	}

	std::sort(gpu_times.begin(), gpu_times.end());
	double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
	std::cout << "Average time per iteration in seconds: " << gpu_avg << "ms" << std::endl;
	std::cout << "Average FPS: " << 1000.0 / gpu_avg << std::endl;
	cin.get();

	return;

	//cv::destroyAllWindows();
}
