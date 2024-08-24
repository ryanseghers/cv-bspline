#include <string>
#include <opencv2/opencv.hpp>

float gaussianf(float x);
float gaussianScaledf(float halfWidth, float height, float center, float x);

cv::Mat ensureImageDims(cv::Mat img, int sizeMult);
cv::Mat loadAndConvertTestImage(const std::string& imgPath, int screenWidth, int screenHeight, bool doConvertTo8u = true, int sizeMult = 64);
