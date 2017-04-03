/* SwapFace class declaration */

#include <opencv2\opencv.hpp>
#include "GlobalParams.h"

class SwapFace {
private:
	cv::Mat fullFrame;
	cv::Mat resizedFrame;

	std::vector<cv::Rect> getFaces();

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

	cv::Mat swapFaces();
};
