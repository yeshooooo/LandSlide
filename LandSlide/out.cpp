#include "out.h"

using namespace cv;

out::out(const std::string &filename)
{
	fs.open(filename, FileStorage::WRITE);
	pointNum = 0;
}

out::~out()
{
	fs.release();
}

cv::FileStorage & out::named()
{
	std::string rockName;
	std::stringstream ss;
	ss << "rock" << pointNum++;
	ss >> rockName;
	fs << rockName;
	return fs;
	// TODO: 在此处插入 return 语句
}
