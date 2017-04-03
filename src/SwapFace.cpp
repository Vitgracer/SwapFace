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

///////////////////////////////////////
// Brief description: 
// Find sking region and postprocess it 
///////////////////////////////////////
cv::Mat SwapFace::findMask(cv::Mat face) {
	int srcW = face.cols;
	int srcH = face.rows;
	cv::resize(face, face, cv::Size(srcW / 4, srcH / 4), 0, 0, cv::INTER_NEAREST);

	cv::cvtColor(face, face, cv::COLOR_BGR2YCrCb);
	cv::Mat bgr[3];
	split(face, bgr);
	bgr[0] = 0;
	cv::merge(bgr, 3, face);

	cv::Mat resMask = segmentFace(face);

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

	cv::resize(resMask, resMask, cv::Size(srcW, srcH), 0, 0, cv::INTER_NEAREST);
	cv::GaussianBlur(resMask, resMask, cv::Size(11, 11), 5, 5);
	cv::threshold(resMask, resMask, 128, 255, cv::THRESH_BINARY);

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
cv::Mat SwapFace::copySrcToDstUsingMask(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::Mat result = imgDst.clone();

	for (int y = 0; y < imgSrc.rows; y++) {
		for (int x = 0; x < imgSrc.cols; x++) {
			if (maskDst.at<uchar>(y, x) == 255) {
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);

				if (maskSrc.at<uchar>(y, x) == 255) {
					result.at<cv::Vec3b>(y, x) = imgSrc.at<cv::Vec3b>(y, x);
				}
			}
		}
	}
	return result;
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

/////////////////////////////////
// Brief description: 
// color interpolation procedure 
/////////////////////////////////
cv::Scalar SwapFace::interpolateColor(cv::Point p, cv::Point closestToSrc, cv::Point closestToDst, cv::Vec3b colorSrc, cv::Vec3b colorDst) {
	cv::Scalar color;
	for (int channel = 0; channel < 3; channel++) {
		float dist = getDist(closestToDst, closestToSrc);
		float distToSrc = getDist(p, closestToSrc);
		float distToDst = getDist(closestToDst, p);
		float colorDist = abs(colorSrc.val[channel] - colorDst.val[channel]);

		color[channel] = colorSrc.val[channel];
	}

	return color;
}

//////////////////////
// Brief description: 
// inpainting 
//////////////////////
cv::Mat SwapFace::stretchFace(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::resize(imgSrc, imgSrc, imgDst.size(), cv::INTER_NEAREST);
	cv::resize(maskSrc, maskSrc, maskDst.size(), cv::INTER_NEAREST);
	maskSrc *= 255;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
	morphologyEx(maskSrc, maskSrc, cv::MORPH_ERODE, element);
	morphologyEx(maskDst, maskDst, cv::MORPH_ERODE, element);

	//cv::Mat fitted = fitOneImgToAnother(imgSrc, imgDst, maskSrc, maskDst);

	cv::Mat cuttedFace = cv::Mat(imgSrc.size(), CV_8UC1);
	imgSrc.copyTo(cuttedFace, maskSrc);

	cv::Mat stretchedFace = copySrcToDstUsingMask(imgSrc, imgDst, maskSrc, maskDst);

	std::vector<std::vector<cv::Point> > contoursSrc;
	cv::findContours(maskSrc, contoursSrc, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	if (contoursSrc.size() == 0) {
		return imgSrc;
	}

	cv::Mat blackMask = cv::Mat(stretchedFace.size(), CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < stretchedFace.rows; y++) {
		for (int x = 0; x < stretchedFace.cols; x++) {
			cv::Point curPoint(x, y);

			if (stretchedFace.at<cv::Vec3b>(curPoint) == cv::Vec3b(255, 255, 255)) {

				cv::Point closestToSrc = findClosesPoint(curPoint, contoursSrc[0]);
				//cv::Point closestToDst = findClosesPoint(curPoint, contoursDst[0]);

				//cv::Scalar color = interpolateColor(curPoint, closestToSrc, closestToDst, cuttedFace.at<cv::Vec3b>(closestToSrc), imgDst.at<cv::Vec3b>(closestToDst));
				cv::Scalar color = cv::Scalar(cuttedFace.at<cv::Vec3b>(closestToSrc).val[0],
					cuttedFace.at<cv::Vec3b>(closestToSrc).val[1],
					cuttedFace.at<cv::Vec3b>(closestToSrc).val[2]);

				cv::circle(stretchedFace, curPoint, 0, color, -1);
				cv::circle(blackMask, curPoint, 0, cv::Scalar(255), -1);
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

		cv::Mat leftFaceImg = cv::Mat(lpSrc[i], rectFaceL);
		cv::Mat rightFaceImg = cv::Mat(lpSrc[i], rectFaceR);

		cv::resize(leftFaceImg, leftFaceImg, cv::Size(rectFaceR.width, rectFaceR.height));
		cv::resize(rightFaceImg, rightFaceImg, cv::Size(rectFaceL.width, rectFaceL.height));

		cv::Mat combUn = lpSrc[i].clone();
		leftFaceImg.copyTo(combUn(rectFaceR), maskFaceR);
		rightFaceImg.copyTo(combUn(rectFaceL), maskFaceL);

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
	cv::resize(usualStitched, usualStitched, cv::Size(fullFrame.cols, fullFrame.rows));
	resultFrame = usualStitched.clone();
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