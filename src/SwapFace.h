/* SwapFace class declaration */

#include <opencv2\opencv.hpp>
#include "GlobalParams.h"

class SwapFace {
private:
	cv::Mat fullFrame;
	cv::Mat resizedFrame;	

	std::vector<cv::Rect> getFaces();
	std::vector<cv::Mat> buildGaussianPyr(cv::Mat img, int pyrLevel);
	std::vector<std::pair<cv::Rect, cv::Mat>> buildGaussianPyr(cv::Rect rect, cv::Mat mask, int pyrLevel);
	std::vector<cv::Mat> buildLaplPyr(std::vector<cv::Mat> gaussPyr, int pyrLevel);

	void swapFace(cv::Rect rect1, cv::Rect rect2);

public:
	SwapFace(cv::Mat& inputFrame) 
		: fullFrame(inputFrame)
	    , resizedFrame(inputFrame) {
		cv::resize(resizedFrame, resizedFrame, cv::Size(WIDTH, HEIGHT));
	};

	virtual ~SwapFace() {
		fullFrame.release();
		resizedFrame.release();
	};

	cv::Mat run();
};
