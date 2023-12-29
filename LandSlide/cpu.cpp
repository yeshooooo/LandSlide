#include "cpu.h"
#include "out.h"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace cv;
using namespace std;

int callBack()
{
	//vector<int> param(4);
	//param[0] = CAP_PROP_FRAME_WIDTH;
	//param[1] = 360;
	//param[2] = CAP_PROP_FRAME_HEIGHT;
	//param[3] = 240;
	VideoCapture cap(fileName);//, CAP_ANY);//, param);
	if (!cap.isOpened()) // Check for invalid input
	{
		cerr << "\n\nCould not open or find the video" << std::endl;
		return -1;
	}
	Size ratio = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));
	cout << endl << ratio.height << "height\n" << ratio.width << "width\n"; //960*544 1280*720 25fps
	//cout << cap.get(CAP_PROP_FPS) << "FPS\n";
	cout << cap.getBackendName() << endl;

	//compression(cap);
	//cap.set(CAP_PROP_FRAME_WIDTH, 0.5*ratio.width);
	//cap.set(CAP_PROP_FRAME_HEIGHT, 0.5*ratio.height);

	bgSub(cap);
	//Mat src = imread("薄雾浓云愁永昼.png");
	//Erosion(src);
	cap.release();
	destroyAllWindows();
	return 0;
}

static void onShot(int mouseEvent, int x, int y, int, void* data)
{
	if (EVENT_RBUTTONDOWN != mouseEvent)
	{
		return;
	}
	Mat *img = (Mat*)data;
	static int name = 0;
	stringstream ss;
	ss.clear();
	ss << name++ << ".png";
	string fileName;
	ss >> fileName;
	imwrite(fileName, *img);
}

static void onMouse(int event, int x, int y, int /*flags*/, void* pointsPtr)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		vector<Point2f>* point = static_cast<vector<Point2f>*>(pointsPtr);
		point->push_back(Point2f((float)x, (float)y));
		//addRemovePt = true;
	}
}

void bgSub(VideoCapture &cap)
{
	Mat frame, fgMask;
	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(history, varThreshold, false);

	//namedWindow("fgMask");
	//moveWindow("fgMask", 440, 240);
	namedWindow("frame");
	
	//badPoint.x = -1;
	//out badArea("badPoints.yml");
	FileStorage in("badPoints.yml", FileStorage::READ);

	//setMouseCallback("fgMask", onShot, &fgMask);
	//setMouseCallback("frame", onShot, &frame);

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, maxCount, epsilon); //| TermCriteria::EPS

	string showFrame;
	string FPS;
	string trackNum;
	int frameIndex = 0;

	vector<Point2f> corners, nextPts, backcross, badPoints;
	//setMouseCallback("frame", onMouse, &badPoints);
	vector<uchar> status;
	vector<float> err;

	cap.read(frame);
	pBackSub->apply(frame, fgMask);
	erode(fgMask, fgMask, kernel);
	Mat prevImg = fgMask;
	goodFeaturesToTrack(fgMask, corners, maxCorners, qualityLevel, minDistance);

	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < maxCorners; ++i)
	{
		int r = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		colors.push_back(Scalar(r, g, b));
	}
	Mat flow(frame.size(), frame.type(), Scalar(0));
	Mat flow_init(frame.size(), frame.type(), Scalar(0));
	Mat init(fgMask.size(), fgMask.type(), Scalar(255));
	Mat mask(fgMask.size(), fgMask.type(), Scalar(255));
	
	if (in.isOpened())
	{
		vector<Point2f> badPoints_file;
		
		FileNode n = in["badPoints"];                         // Read string sequence - Get node
		//if (n.type() != FileNode::MAP)
		//{
		//	cerr << "badPoints is not a map! FAIL" << endl;
		//	exit(-1);
		//}
		//FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
		//for (; it != it_end; ++it)
		//{
		//	float x = (float)*it[]
		//}
		n >> badPoints_file;
		for (Point2f x : badPoints_file)
		{
			cv::circle(init, x, maskRadius, Scalar(0), -1);
			cv::circle(flow_init, x, maskRadius, Scalar(0, 0, 255), 3);
		}
		in.release();
	}
	vector<vector<Point2f> > tracks;
	vector<vector<Point2f> > tracks_new;
	//vector<Point2f> betterPts(maxCorners);
	out outPoints("outPoints.yml");
	TickMeter tm;
	std::vector<double> gpu_times(1900);
	double temp;

	while (cap.grab())
	{
		//long long start = getTickCount();
		tm.reset();
		tm.start();
		temp = tm.getTimeMilli();
		//main function
		if (0 == frameIndex % retrieveIterval)
		{
			cap.retrieve(frame);
			
			pBackSub->apply(frame, fgMask);
			
			morphologyEx(fgMask, fgMask, operation, kernel);
			
			if (0 < badPoints.size())//过滤干扰区域
			{
				for (Point2f badPoint : badPoints)
				{
					cv::circle(frame, badPoint, maskRadius, Scalar(0, 0, 255), 3);
				}
			}
			if (0 < tracks.size())//主光流过程
			{
				corners.clear();
				tracks_new.clear();

				for (int i = 0; i < tracks.size(); ++i)//init corners
				{
					corners.push_back(tracks[i].back());
				}
				//计算2光流
				calcOpticalFlowPyrLK(prevImg, fgMask, corners, nextPts, status, err, winSize, maxLevel, criteria);
				calcOpticalFlowPyrLK(fgMask, prevImg, nextPts, backcross, status, noArray(), winSize, maxLevel, criteria);
				//sift
				for (int i = 0; i < tracks.size(); ++i)
				{
					if (1 != status[i])
					{
						if(2 < tracks[i].size())
							outPoints.named() << tracks[i];
						continue;
					}
					if (corners[i].y > nextPts[i].y)
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
					
					if (i < maxCorners)
					{
						cv::line(flow, corners[i], nextPts[i], colors[i], 2);
						cv::circle(frame, nextPts[i], 3, colors[i], -1);
					}

				}//sift end
				tracks = tracks_new;
			}//end main
			
			//cv::imshow("fgMask", fgMask);
			prevImg = fgMask.clone();

			//betterPts.clear();
			
			//corners = betterPts;
			cv::add(flow, frame, frame);
			cv::rectangle(frame, Point(10, 2), Point(240, 60), Scalar(255, 255, 255), FILLED);
			//showFrame = to_string(cap.get(CAP_PROP_POS_FRAMES));
			//++frameIndex;
			
			showFrame = "Frame: " + to_string(frameIndex);
			cv::putText(frame, showFrame.c_str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			
		}
		if (0 == frameIndex % detectIterval)//监测角点
		{
			init.copyTo(mask);
			if (0 < nextPts.size())//过滤已有点周围区域
			{
				for (int i = 0; i < tracks.size(); ++i)
				{
					cv::circle(mask, nextPts[i], 5, Scalar(0), -1);
				}
			}
			if (0 < badPoints.size())//过滤干扰区域
			{
				for (Point2f badPoint : badPoints)
				{
					cv::circle(mask, badPoint, maskRadius, Scalar(0), -1);
				}
			}

			goodFeaturesToTrack(fgMask, corners, maxCorners, qualityLevel, minDistance, mask);
			for (Point2f x : corners)
			{
				vector<Point2f> tmp{ x };
				tracks.push_back(tmp);
			}
		}
		if (0 == frameIndex % clearIterval)
		{
			flow = flow_init.clone();
		}
		frameIndex = int(cap.get(CAP_PROP_POS_FRAMES));
		trackNum = "Tracking: " + to_string(tracks.size());
		cv::putText(frame, trackNum.c_str(), Point(15, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		tm.stop();

		//int fps = (int)(getTickFrequency() / (getTickCount() - start));

		//FPS = "FPS: " + to_string(fps);
		gpu_times.push_back(tm.getTimeMilli());
		FPS = "Frame generation time:" + to_string(int(tm.getTimeMilli() - temp)) + "ms";
		cv::putText(frame, FPS.c_str(), Point(15, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		cv::imshow("frame", frame);

		int keyboard = waitKey(waitTime);
		if (27 == keyboard)
			break;
		switch (keyboard)
		{
		case 's': 
		{
			FileStorage out("badPoints.yml", FileStorage::WRITE);
			out << "badPoints" << badPoints;
			out.release();
			break;
		}
		case 'c':
			flow = flow_init.clone();
			break;
		default:
			break;

		}
	}
	std::sort(gpu_times.begin(), gpu_times.end());
	double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
	std::cout << "Average time per iteration in seconds: " << gpu_avg << "ms" << std::endl;
	std::cout << "Average FPS: " << 1000.0 / gpu_avg << std::endl;
	cin.get();
	return;
}


static void onErode(int pos, void* data)
{
	Mat *image = static_cast<Mat *>(data);
	//Mat *image = (Mat*)data;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * pos + 1, 2 * pos + 1));
	Mat dst;
	erode(*image, dst, kernel);
	cv::imshow("Erode", dst);
}


void Erosion(Mat &src)
{
	namedWindow("Erotion");
	const int maxKernel = 21;
	int erosion_size = 0;
	//createTrackbar("0:Rect \n 1: Cross \n 2: Ellipse", "Erotion", &element, maxElem);
	createTrackbar("Kernel size:\n 2n +1", "Erotion", &erosion_size, maxKernel, onErode, &src);
	waitKey();
	//switch (element)
	//{
	//case 0: type = MORPH_RECT; break;
	//case 1: type = MORPH_CROSS; break;
	//case 2: type = MORPH_ELLIPSE; break;
	//default: type = MORPH_ELLIPSE; break;
	//}
}

void compression(VideoCapture &cap)
{
	const string outName = "compress2.avi";
	double fps = cap.get(CAP_PROP_FPS);
	int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');
	//Size s = Size(600, 400);
	Size ratio = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));
	VideoWriter writer;
	writer.open(outName, ex, fps, ratio);
	Mat src;
	do
	{
		cap.read(src);
		if (src.empty()) break;
		//imshow("src", src);
		writer.write(src);
		//int keyboard = waitKey(1);
		//if (27 <= keyboard)
		//	break;
	} while (true);

}