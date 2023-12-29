#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;

//帧差法检测车辆
Mat MoveCheck(Mat& frontMat, Mat& afterMat)
{
	Mat frontGray, afterGray, diffGray;
	Mat resframe = afterMat.clone();
	//灰度处理
	cvtColor(frontMat, frontGray, CV_BGR2GRAY);
	cvtColor(afterMat, afterGray, CV_BGR2GRAY);
	//imshow("GRAY", frontGray);

	//帧差处理 找到帧与帧之间运动的物体差异
	//缺点：会把其他运动物体也算进来
	absdiff(frontGray, afterGray, diffGray);
	//imshow("absdiff", diffGray);

	//二值化：黑白分明 会产生大量白色噪点
	threshold(diffGray, diffGray, 25, 255, CV_THRESH_BINARY);
	//imshow("diff", diffGray);

	//腐蚀处理：去除白色噪点 噪点不能完全去除，反而主要物体会被腐蚀的图案都变得不明显
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(diffGray, diffGray, element);
	//imshow("erode", diffGray);

	//膨胀处理：将白色区域变“胖”
	Mat element2 = cv::getStructuringElement(MORPH_RECT, Size(20, 20));
	dilate(diffGray, diffGray, element2);
	//imshow("dilate", diffGray);

	//动态物体标记
	vector<vector<Point>> contours; //保存关键点
	findContours(diffGray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//提取关键点
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	int x, y, w, h;
	int num = contours.size();

	for (int i = 0; i < num; i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

		x = boundRect[i].x;
		y = boundRect[i].y;
		w = boundRect[i].width;
		h = boundRect[i].height;

		//绘制
		rectangle(resframe, Point(x, y), Point(x + w, y + h), Scalar(0, 255, 0), 4);
	}
	return resframe;
}

int main(int argc, char* argv[])
{
	Mat frame;
	Mat tempframe;
	Mat res;
	int count = 0;

	VideoCapture cap("stone.mov");
	while (cap.read(frame))
	{
		count++;
		if (count == 1)
		{
			res = MoveCheck(frame, frame);
		}
		else
		{
			res = MoveCheck(tempframe, frame);
		}
		tempframe = frame.clone();
		imshow("frame", frame);//原视频帧
		imshow("res", res);//框选后的视频帧
		waitKey(25);
	}
	return 0;
}