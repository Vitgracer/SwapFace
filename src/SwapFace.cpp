/* Implementation of TextRecognition class methods*/

#include "TextRecognizer.h"
#include <algorithm>

TextRecognizer::~TextRecognizer() {}

///////////////////////////////////////////////////////////////////////
// Brief description: 
// 1. get input src frame
// 2. perform gradient morphology to find high intensity edges
// 3. perform otsu thresholding to extract text
// 4. Perform median blur to remove salt=pepper noise 
// 5. Perform dilation only in vertical direction to avoid test gluing 
// 6. Return processed image
/////////////////////////////////////////////////////////////////////// 
cv::Mat TextRecognizer::gainTextRegions(cv::Mat& frame) {
	cv::Mat grayFrame;
	cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

	cv::Mat gradientImage;
	cv::Mat morphKernelGrad = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(grayFrame, gradientImage, cv::MORPH_GRADIENT, morphKernelGrad);

	cv::Mat thresholdedImage;
	cv::threshold(gradientImage, thresholdedImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	cv::medianBlur(thresholdedImage, thresholdedImage, 5);

	cv::Mat morphKernelClose = getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	cv::morphologyEx(thresholdedImage, thresholdedImage, CV_MOP_CLOSE, morphKernelClose);

	cv::Mat morphKernelDil = getStructuringElement(cv::MORPH_RECT, cv::Size(COMPOUND_TEXT_PARAMETER, 3));
	cv::morphologyEx(thresholdedImage, thresholdedImage, CV_MOP_DILATE, morphKernelDil);

	return thresholdedImage;
}


/////////////////////////////////////////////
// Brief description: 
// sort vector of rects by rects y-coordinate 
/////////////////////////////////////////////
bool compareByMC(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
	cv::Rect boundingBoxA = cv::boundingRect(a);
	cv::Rect boundingBoxB = cv::boundingRect(b);

	return boundingBoxA.y < boundingBoxB.y;
}

/////////////////////////////////////////////
// Brief description: 
// 1. get input processed frame
// 2. find contours
// 3. find bounding boxes for this contours 
// 4. Return vector with bounding boxes 
/////////////////////////////////////////////
void TextRecognizer::findTextStringsBoundingRects(cv::Mat& frame) {
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(frame, contours, 0, 1);

#if VISUALIZATION
	cv::Mat contoursVisualization;
	cv::cvtColor(frame, contoursVisualization, CV_GRAY2BGR);
	drawContours(contoursVisualization, contours, -1, cv::Scalar(255, 255, 0), 3);
#endif 

	std::sort(contours.begin(), contours.end(), compareByMC);
	
	for (int i = 0; i < contours.size(); i++) {
		textStringsBoundingRects.push_back(std::make_pair(cv::minAreaRect(cv::Mat(contours[i])), cv::boundingRect(contours[i])));
	}

	alignTextStrings();
}

///////////////////////////////////////////////////////////////////////////////
// Brief description: 
// 1. Use input fullFrame and text string bounding rects 
// 2. Rotate each string Matrix using angle of rotation rect
// 3. Write result to the class member std::vector<cv::Mat> alignedTextStrings
///////////////////////////////////////////////////////////////////////////////
void TextRecognizer::alignTextStrings() {
	for (int i = 0; i < textStringsBoundingRects.size(); i++) {
		cv::Mat oneString = cv::Mat(fullFrame, textStringsBoundingRects[i].second);		
		
		double angle = textStringsBoundingRects[i].first.angle;
		cv::Point rotateCenter = textStringsBoundingRects[i].first.center;
		
		cv::Mat transformMatrix = cv::getRotationMatrix2D(rotateCenter, angle, 1);
		cv::Mat alignedString;
		cv::warpAffine(oneString, alignedString, transformMatrix, oneString.size());
		
		alignedTextStrings.push_back(alignedString);
	}
}

/////////////////////////////////////////////
// Brief description: 
// Preprocess cv::Mat aligned text string to 
// use Tesseract
/////////////////////////////////////////////
cv::Mat TextRecognizer::processTextString(cv::Mat textStringMat) {
	cv::Mat grayFrame;
	cv::cvtColor(textStringMat, grayFrame, CV_BGR2GRAY);

	cv::Point blackPoint(0, 0);
	for (int y = 0; y < grayFrame.rows; y++) {
		uchar* data = grayFrame.ptr<uchar>(y);

		for (int x = 0; x < grayFrame.cols; x++) {
			if (data[x] == 0) {
				blackPoint = cv::Point(x, y);
				goto ex;
			}
		}
	}

ex:
	if (blackPoint == cv::Point(0, 0)) {
		double meanVal = cv::mean(grayFrame)[0];
		cv::floodFill(grayFrame, blackPoint, (int)meanVal);
	}
	
	cv::GaussianBlur(grayFrame, grayFrame, cv::Size(3, 3), 0.5);
	cv::threshold(grayFrame, grayFrame, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

	cv::Mat morphKernelOpen = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(grayFrame, grayFrame, CV_MOP_OPEN, morphKernelOpen);

	return grayFrame;
}

/////////////////////////////////////////////
// Brief description: 
// 1. Preprocess cv::Mat aligned text string 
// 2. Use Tesseract to recognize
/////////////////////////////////////////////
std::string TextRecognizer::recognizeTextString() {
	for (int i = 0; i < alignedTextStrings.size(); i++) {
		cv::Mat processedString = processTextString(alignedTextStrings[i]);
	}

	return std::string();
}

//////////////////////////////////
// Brief description: 
// public funcion to run full code
//////////////////////////////////
void TextRecognizer::run() {
	auto gainedTextRegions = gainTextRegions(fullFrame);
	findTextStringsBoundingRects(gainedTextRegions);
	std::string recognizedText = recognizeTextString();

#if VISUALIZATION
	cv::Mat boundingRectsVisualization = fullFrame.clone();	

	for (int i = 0; i < textStringsBoundingRects.size(); i++) {
		cv::Point2f rotatedRectPoints[4]; 
		textStringsBoundingRects[i].first.points(rotatedRectPoints);

		for (int j = 0; j < 4; j++)
			line(boundingRectsVisualization, rotatedRectPoints[j], rotatedRectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 3);
	}

	cv::imshow("Found Text", boundingRectsVisualization);
	bool finish = true;
#endif 	
}