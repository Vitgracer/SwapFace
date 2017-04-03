/* SwapFace class declaration */

#include <opencv2\opencv.hpp>
#include "GlobalParams.h"

class SwapFace {
private:
	cv::Mat fullFrame;

public:
	SwapFace(cv::Mat& inputFrame) 
		: fullFrame(inputFrame) {};

	virtual ~SwapFace() {};

	cv::Mat swapFaces();
};
