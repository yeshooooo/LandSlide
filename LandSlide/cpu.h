#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

const std::string fileName = "stone.mov";

const int history = 150;
const double varThreshold = 16.0;
//const double dist2Threshhold = 400.0;

const int operation = cv::MORPH_OPEN;
const int maxCount = 10;
const double epsilon = 0.03;

const int clearIterval = 250;
const int detectIterval = 5;
const int retrieveIterval = 1;

const int maxCorners = 15;
const double qualityLevel = 0.25;
const double minDistance = 25.0;

const cv::Size winSize(cv::Size(21, 21));
const int maxLevel(3);
const int errorMax(4);
const int maskRadius{ 45 };
const int waitTime{ 10 };

static void onMouse(int mouseEvent, int x, int y, int, void* data);

static void onErode(int pos, void* data);

int callBack();

void Erosion(cv::Mat &src);

void compression(cv::VideoCapture &cap);

void bgSub(cv::VideoCapture &cap);


