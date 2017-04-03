/* Implementation of SwapFace class methods*/

#include "SwapFace.h"

////////////////////////////////////////
// Brief description: 
// find faces usinng available cascades
///////////////////////////////////////
std::vector<cv::Rect> SwapFace::getFaces() {
	cv::CascadeClassifier face_cascade;
	face_cascade.load(CASCADE_PATH);
	
	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(resizedFrame, faces, 
								  SCALE_FACTOR, MIN_NEIGHBOURS, 
								  2, cv::Size(WINDOW_SIZE, WINDOW_SIZE));

#if VISUALIZATION
	cv::Mat facesVisualization = resizedFrame.clone();
	for (int i = 0; i < faces.size(); i++) {
		cv::rectangle(facesVisualization, faces[i], cv::Scalar(255, 0, 255));
	}
	cv::imshow("detected faces", facesVisualization);
#endif 

	return faces;
}

////////////////////////////////////////
// Brief description: 
// function to build gaussian pyramid, 
// when each image is resized in x2
///////////////////////////////////////
std::vector<cv::Mat> SwapFace::buildGaussianPyr(cv::Mat img, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<cv::Mat> gaussPyr(0, cv::Mat());
	cv::Mat downImg = img.clone();
	gaussPyr.push_back(downImg);

	for (int i = 0; i < pyrLevel; i++) {
		cv::pyrDown(downImg, downImg);
		gaussPyr.push_back(downImg);
	}

	return gaussPyr;
}

////////////////////////////////////////////
// Brief description: 
// function to build gaussian pyramid, 
// when each image is resized in x2.
// Overloaded function to get resized rects 
///////////////////////////////////////////
std::vector<std::pair<cv::Rect, cv::Mat>> SwapFace::buildGaussianPyr(cv::Rect rect, cv::Mat mask, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<std::pair<cv::Rect, cv::Mat> > gaussPyr(0, std::make_pair(cv::Rect(), cv::Mat()));
	cv::Rect downRect = rect;
	cv::Mat maskRect;
	cv::resize(mask, maskRect, cv::Size(downRect.width, downRect.height));

	gaussPyr.push_back(std::make_pair(downRect, maskRect));

	for (int i = 0; i < pyrLevel; i++) {
		downRect.x = downRect.x / 2;
		downRect.y = downRect.y / 2;
		downRect.width = downRect.width / 2;
		downRect.height = downRect.height / 2;
		cv::resize(maskRect, maskRect, cv::Size(downRect.width, downRect.height));
		gaussPyr.push_back(std::make_pair(downRect, maskRect));
	}

	return gaussPyr;
}

//////////////////////////////////////////////////////
// Brief description: 
// function to build laplacian pyramid, 
// when each image is obtained by substraction of real
// image and its resized analog
//////////////////////////////////////////////////////
std::vector<cv::Mat> SwapFace::buildLaplPyr(std::vector<cv::Mat> gaussPyr, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<cv::Mat> laplPyr(0, cv::Mat());
	laplPyr.push_back(gaussPyr[pyrLevel]);

	cv::Mat upImg;
	cv::Mat laplMat;

	for (int i = pyrLevel; i >= 1; i--) {
		cv::pyrUp(gaussPyr[i], upImg);
		laplMat = gaussPyr[i - 1] - upImg;
		laplPyr.push_back(laplMat);
	}

	return laplPyr;
}

///////////////////////////////////////
// Brief description: 
// KMeans using to extract skin region 
///////////////////////////////////////
cv::Mat SwapFace::segmentFace(cv::Mat src) {
	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++) {

		for (int x = 0; x < src.cols; x++) {
			for (int z = 0; z < 3; z++) {
				samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y, x)[z];
			}
		}
	}

	cv::Mat centers;
	cv::Mat labels;
	kmeans(samples, CLUSTER_COUNT, labels, 
		   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), 
		   CLUSTER_ATTEMPTS, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat resMask(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++) {
		uchar* dataMask = resMask.data + resMask.step.buf[0] * y;

		for (int x = 0; x < src.cols; x++) {
			dataMask[x] = (uchar)labels.at<int>(y + x*src.rows, 0) * 255;
		}
	}

	// check if inverse needed
	int whites = 0;
	int blacks = 0;

	cv::Mat ellipse = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(ellipse, cv::RotatedRect(cv::Point(ellipse.cols / 2, ellipse.rows / 2), cv::Size(3 * ellipse.cols / 4, ellipse.rows), 0), cv::Scalar(255), -1);

	for (int y = 0; y < src.rows; y++) {
		uchar* dataMask = resMask.data + resMask.step.buf[0] * y;
		uchar* dataEllipse = ellipse.data + ellipse.step.buf[0] * y;

		for (int x = 0; x < src.cols; x++) {
			if (dataEllipse[x] == 255) {
				if (dataMask[x] == 255) whites++;
				else blacks++;
			}
		}
	}

	if (blacks > whites) resMask = 255 - resMask;

	return resMask;
}

cv::Mat performHistAnalysis(cv::Mat& src, cv::Mat mask, const int delta = 40) {
	int bins = 256;

	int histSize[] = { bins };
	float range[] = { 0, 256 };

	const float* ranges[] = { range };
	cv::MatND hist;

	int channels[] = { 0 };

	cv::calcHist(&src, 1, channels, mask,
		hist, 1, histSize, ranges,
		true,
		false);

	double maxVal = 0;
	cv::Point maxP(0, 0);
	cv::minMaxLoc(hist, 0, &maxVal, 0, &maxP);

	cv::Mat maskChannel1, maskChannel2, maskUpd;
	cv::threshold(src, maskChannel1, maxP.y - delta, 255, cv::THRESH_BINARY);
	cv::threshold(src, maskChannel2, maxP.y + delta, 255, cv::THRESH_BINARY_INV);
	cv::bitwise_and(maskChannel1, maskChannel2, maskUpd);

	return maskUpd;
}

cv::Mat processMask(cv::Mat src) {
	cv::Mat mask = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(mask, cv::RotatedRect(cv::Point(mask.cols / 2, mask.rows / 2), 
				      cv::Size(mask.cols / 2, 3 * mask.rows / 4), 0), cv::Scalar(255), -1);

	cv::Mat hsvSRC;
	cv::cvtColor(src, hsvSRC, cv::COLOR_RGB2HSV);
	cv::Mat bgr[3];
	split(hsvSRC, bgr);

	cv::Mat mask1 = performHistAnalysis(bgr[0], mask);
	cv::Mat mask2 = performHistAnalysis(bgr[1], mask);
	cv::Mat mask3 = performHistAnalysis(bgr[2], mask);

	cv::Mat combMask;
	cv::bitwise_and(mask1, mask2, mask2);
	cv::bitwise_and(mask2, mask3, combMask);

	return combMask;
}

///////////////////////////////////////
// Brief description: 
// Find sking region and postprocess it 
///////////////////////////////////////
cv::Mat SwapFace::findMask(cv::Mat face) {
	int srcW = face.cols;
	int srcH = face.rows;
	cv::resize(face, face, cv::Size(srcW / 4, srcH / 4), 0, 0, cv::INTER_NEAREST);

	cv::Mat resMask = segmentFace(face);
	//cv::Mat updMask = processMask(face);
	//cv::bitwise_and(resMask, updMask, resMask);

	cv::resize(resMask, resMask, cv::Size(srcW, srcH), 0, 0, cv::INTER_NEAREST);
	cv::GaussianBlur(resMask, resMask, cv::Size(11, 11), 5, 5);
	cv::threshold(resMask, resMask, 128, 255, cv::THRESH_BINARY);

	int maxLength = 0;
	int maxInd = 0;

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(resMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() == 0)
		return resMask;

	for (int i = 0; i < contours.size(); i++) {
		double length = contours[i].size();
		if (length > maxLength) {
			maxLength = length;
			maxInd = i;
		}
	}

	resMask = 0;
	drawContours(resMask, contours, maxInd, cv::Scalar(255), CV_FILLED);

	std::vector<cv::Point> hull;
	cv::convexHull(contours[maxInd], hull);

	if (hull.size() == 0) {
		return resMask;
	}

	resMask = 0;
	auto hullRes = std::vector<std::vector<cv::Point> >(1, hull);
	drawContours(resMask, hullRes, 0, cv::Scalar(255), -1);

	cv::Mat restrictionEllipse = cv::Mat(resMask.rows, resMask.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(restrictionEllipse, cv::RotatedRect(cv::Point(resMask.cols / 2, resMask.rows / 2), cv::Size(resMask.cols - 6, resMask.rows - 6), 0), cv::Scalar(255), -1);
	cv::bitwise_and(restrictionEllipse, resMask, resMask);

	return resMask;
}

///////////////////////////////////////
// Brief description: 
// Function to copy one face to another 
// with some restrictions 
///////////////////////////////////////
std::pair<cv::Mat, cv::Mat> SwapFace::copySrcToDstUsingMask(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::Mat result = imgDst.clone();
	cv::Mat blackMask = cv::Mat(result.size(), CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < imgSrc.rows; y++) {
		for (int x = 0; x < imgSrc.cols; x++) {
			if (maskDst.at<uchar>(y, x) == 255) {
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);

				if (maskSrc.at<uchar>(y, x) == 255) {
					result.at<cv::Vec3b>(y, x) = imgSrc.at<cv::Vec3b>(y, x);
				}
				else {
					blackMask.at<uchar>(y, x) = 255;
				}
			}
		}
	}

	return std::make_pair(result, blackMask);
}

///////////////////////////////////////
// Brief description: 
// get distance between two points 
///////////////////////////////////////
float SwapFace::getDist(cv::Point& p1, cv::Point& p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

///////////////////////////////////////////
// Brief description: 
// get closest point to point P in contour 
///////////////////////////////////////////
cv::Point SwapFace::findClosesPoint(cv::Point& p, std::vector<cv::Point>& contour) {
	float mindist = 10000;
	int minInd = 0;

	for (int i = 0; i < contour.size(); i++) {
		cv::Point contourPoint = contour[i];

		float dist = getDist(contourPoint, p);
		if (dist < mindist) {
			mindist = dist;
			minInd = i;
		}
	}

	return contour[minInd];
}

//////////////////////
// Brief description: 
// inpainting 
//////////////////////
cv::Mat SwapFace::stretchFace(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::resize(imgSrc, imgSrc, imgDst.size(), cv::INTER_NEAREST);
	cv::resize(maskSrc, maskSrc, maskDst.size(), cv::INTER_NEAREST);
	maskSrc *= 255;

	cv::Mat cuttedFace = cv::Mat(imgSrc.size(), CV_8UC1);
	imgSrc.copyTo(cuttedFace, maskSrc);

	auto stretched = copySrcToDstUsingMask(imgSrc, imgDst, maskSrc, maskDst);
	cv::Mat stretchedFace = stretched.first;
	cv::Mat blackMask = stretched.second;

	std::vector<std::vector<cv::Point> > contoursSrc;
	cv::findContours(maskSrc, contoursSrc, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	if (contoursSrc.size() == 0) {
		return imgSrc;
	}

	for (int y = 0; y < stretchedFace.rows; y++) {
		for (int x = 0; x < stretchedFace.cols; x++) {
			cv::Point curPoint(x, y);

			if (blackMask.at<uchar>(y, x) == 255) {

				cv::Point closestToSrc = findClosesPoint(curPoint, contoursSrc[0]);
				
				cv::Scalar color = cv::Scalar(cuttedFace.at<cv::Vec3b>(closestToSrc).val[0],
					cuttedFace.at<cv::Vec3b>(closestToSrc).val[1],
					cuttedFace.at<cv::Vec3b>(closestToSrc).val[2]);

				cv::circle(stretchedFace, curPoint, 0, color, -1);
			}
		}
	}

	cv::Mat stretchedClone = stretchedFace.clone();
	cv::GaussianBlur(stretchedClone, stretchedClone, cv::Size(11, 11), 3, 3);
	stretchedClone.copyTo(stretchedFace, blackMask);

	return stretchedFace;
}

///////////////////////////////////////////
// Brief description: 
// fit left image to right with right mask 
// and the same for the right image 
//////////////////////////////////////////
std::pair<cv::Mat, cv::Mat> SwapFace::fitImagesToEachOther(cv::Mat leftImg, cv::Mat rightImg, cv::Mat leftMask, cv::Mat rightMask) {
	cv::Mat leftFaceFitted = stretchFace(leftImg, rightImg, leftMask, rightMask);
	cv::Mat rightFaceFitted = stretchFace(rightImg, leftImg, rightMask, leftMask);

	return std::make_pair(leftFaceFitted, rightFaceFitted);
}

////////////////////////////////
// Brief description: 
// main algorithm to swap faces 
////////////////////////////////
void SwapFace::swapFace(cv::Rect lFace, cv::Rect rFace) {
	cv::Mat leftFaceImg = cv::Mat(resizedFrame, lFace);
	cv::Mat rightFaceImg = cv::Mat(resizedFrame, rFace);

	//leftFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l1.png");
	//rightFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l2.png");
	cv::Mat leftMask = findMask(leftFaceImg);
	cv::Mat rightMask = findMask(rightFaceImg);

	auto fittedImages = fitImagesToEachOther(leftFaceImg, rightFaceImg, leftMask, rightMask);

	cv::Mat comb = resizedFrame.clone();
	fittedImages.first.copyTo(comb(rFace), rightMask);
	fittedImages.second.copyTo(comb(lFace), leftMask);

	resizedFrame.convertTo(resizedFrame, CV_32F, 1.0 / 255.0);
	comb.convertTo(comb, CV_32F, 1.0 / 255.0);

	auto gpSrc = buildGaussianPyr(resizedFrame);
	auto lpSrc = buildLaplPyr(gpSrc);

	auto gpComb = buildGaussianPyr(comb);
	auto lpComb = buildLaplPyr(gpComb);

	auto gpRectL = buildGaussianPyr(lFace, leftMask);
	auto gpRectR = buildGaussianPyr(rFace, rightMask);

	std::reverse(gpRectL.begin(), gpRectL.end());
	std::reverse(gpRectR.begin(), gpRectR.end());

	std::vector<cv::Mat> laplStitched;
	laplStitched.push_back(lpComb[0]);

	for (int i = 1; i < lpSrc.size(); i++) {
		cv::Rect rectFaceL = gpRectL[i].first;
		cv::Rect rectFaceR = gpRectR[i].first;

		cv::Mat maskFaceL = gpRectL[i].second;
		cv::Mat maskFaceR = gpRectR[i].second;
		maskFaceL *= 255;
		maskFaceR *= 255;

		cv::Mat leftFaceImg1 = cv::Mat(lpComb[i], rectFaceR);
		cv::Mat rightFaceImg1 = cv::Mat(lpComb[i], rectFaceL);

		cv::Mat combUn1 = lpSrc[i].clone();
		cv::Mat element2;
		if (i == 1) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		if (i == 2) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
		if (i == 3) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
		if (i == 4) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));

		morphologyEx(maskFaceR, maskFaceR, cv::MORPH_ERODE, element2);
		morphologyEx(maskFaceL, maskFaceL, cv::MORPH_ERODE, element2);

		leftFaceImg1.copyTo(combUn1(rectFaceR), maskFaceR);
		rightFaceImg1.copyTo(combUn1(rectFaceL), maskFaceL);

		laplStitched.push_back(combUn1);
	}

	cv::Mat usualStitched = laplStitched[0].clone();
	for (int i = 1; i <= 4; i++) {
		cv::Mat up;
		cv::pyrUp(usualStitched, up);
		usualStitched = up + laplStitched[i];
	}
	usualStitched.convertTo(usualStitched, CV_8UC3, 255);
	cv::resize(usualStitched, resultFrame, cv::Size(fullFrame.cols, fullFrame.rows));	
}

//////////////////////////////////
// Brief description: 
// public funcion to run full code
//////////////////////////////////
cv::Mat SwapFace::run() {
	auto faces = getFaces();

	if (faces.size() != 2) 
		return resizedFrame;
	
	cv::Rect lFace = faces[0];
	cv::Rect rFace = faces[1];

	if (faces[0].x > faces[1].x) {
		lFace = faces[1];
		lFace = faces[0];
	}	

	swapFace(lFace, rFace);

	return resultFrame;
}