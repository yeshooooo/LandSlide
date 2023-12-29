#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;

//֡���⳵��
Mat MoveCheck(Mat& frontMat, Mat& afterMat)
{
	Mat frontGray, afterGray, diffGray;
	Mat resframe = afterMat.clone();
	//�Ҷȴ���
	cvtColor(frontMat, frontGray, CV_BGR2GRAY);
	cvtColor(afterMat, afterGray, CV_BGR2GRAY);
	//imshow("GRAY", frontGray);

	//֡��� �ҵ�֡��֮֡���˶����������
	//ȱ�㣺��������˶�����Ҳ�����
	absdiff(frontGray, afterGray, diffGray);
	//imshow("absdiff", diffGray);

	//��ֵ�����ڰ׷��� �����������ɫ���
	threshold(diffGray, diffGray, 25, 255, CV_THRESH_BINARY);
	//imshow("diff", diffGray);

	//��ʴ����ȥ����ɫ��� ��㲻����ȫȥ����������Ҫ����ᱻ��ʴ��ͼ������ò�����
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(diffGray, diffGray, element);
	//imshow("erode", diffGray);

	//���ʹ�������ɫ����䡰�֡�
	Mat element2 = cv::getStructuringElement(MORPH_RECT, Size(20, 20));
	dilate(diffGray, diffGray, element2);
	//imshow("dilate", diffGray);

	//��̬������
	vector<vector<Point>> contours; //����ؼ���
	findContours(diffGray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//��ȡ�ؼ���
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

		//����
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
		imshow("frame", frame);//ԭ��Ƶ֡
		imshow("res", res);//��ѡ�����Ƶ֡
		waitKey(25);
	}
	return 0;
}