#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    cv::Mat plotPointsAndCurve(const std::string& chartTitle, const std::vector<cv::Point2f>& points, const std::vector<cv::Point2f>& curvePoints);
    cv::Mat plotPointsAndCurve(const std::string& chartTitle, const std::vector<float>& pointValues, const std::vector<cv::Point2f>& curvePoints);
    cv::Mat plotTwoCurves(const std::string& chartTitle, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);
    void tryCvPlot();
}
