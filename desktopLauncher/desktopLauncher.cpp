#include "SwapFace.h"

int main() {
#ifdef VIDEO_MODE
	cv::VideoCapture cap;

	if (!cap.open(CAMERA_INDEX))
		return 0;

	while (true) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		cv::imshow("input", frame);

		SwapFace faceSwapper = SwapFace(frame);
		cv::Mat swappedFaces = faceSwapper.swapFaces();
		cv::imshow("swappedFaces", swappedFaces);
		
		// ESC to stop
		if (cv::waitKey(1) == 27) break;
	}
#else
	cv::Mat inputFrame = cv::imread(IMAGE_PATH);

	SwapFace faceSwapper = SwapFace(inputFrame);
	cv::Mat swappedFaces = faceSwapper.run();

#endif
	return 0;
}