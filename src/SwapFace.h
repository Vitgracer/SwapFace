/* TextRecognition class declaration */

#include <opencv2\opencv.hpp>
#include "GlobalParams.h"

class TextRecognizer {
private:
	cv::Mat fullFrame;
	std::vector<std::pair<cv::RotatedRect, cv::Rect>> textStringsBoundingRects;
	std::vector<cv::Mat> alignedTextStrings;

	cv::Mat gainTextRegions(cv::Mat& frame);
	void findTextStringsBoundingRects(cv::Mat& frame);	
	void alignTextStrings();
	cv::Mat processTextString(cv::Mat textStringMat);
	std::string recognizeTextString();

public:
	TextRecognizer(cv::Mat& inputFrame) 
		: fullFrame(inputFrame)
		, textStringsBoundingRects(0, std::make_pair(cv::RotatedRect(), cv::Rect()))
	    , alignedTextStrings(0, cv::Mat()) {};

	virtual ~TextRecognizer();

	void run();
};

// additional functions 
bool compareByMC(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);