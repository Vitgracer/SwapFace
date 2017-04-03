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

		SwapFace engine = SwapFace(frame);
		engine.run();

		// ESC to stop
		if (cv::waitKey(1) == 27) break;
	}
#else
	cv::Mat inputFrame = cv::imread(IMAGE_PATH);

	SwapFace engine = SwapFace(inputFrame);
	engine.run();
#endif
	return 0;
}