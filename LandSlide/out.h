#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <iostream>

class out
{
	int pointNum;
public:

	out(const std::string &filename);
	~out();
	cv::FileStorage& named();

	cv::FileStorage fs;
};
