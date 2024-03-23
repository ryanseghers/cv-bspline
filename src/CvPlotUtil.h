#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    cv::Mat plotPointsAndCurve(const std::string& chartTitle, const std::vector<cv::Point2f>& points, const std::vector<cv::Point2f>& curvePoints);
    void tryCvPlot();
}
