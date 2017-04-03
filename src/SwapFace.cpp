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
	face_cascade.detectMultiScale(resizedFrame, faces, 1.1, 2, 2, cv::Size(100, 100));

#if VISUALIZATION
	cv::Mat facesVisualization = resizedFrame.clone();
	for (int i = 0; i < faces.size(); i++) {
		cv::rectangle(facesVisualization, faces[i], cv::Scalar(255, 0, 255));
	}
#endif 

	return faces;
}

//////////////////////////////////
// Brief description: 
// public funcion to run full code
//////////////////////////////////
cv::Mat SwapFace::swapFaces() {
	auto faces = getFaces();

	return cv::Mat();
}