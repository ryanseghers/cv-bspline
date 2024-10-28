#pragma once
#include <string>
#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    cv::Mat plotPointsAndCurve(const std::string& chartTitle, const std::vector<cv::Point2f>& points, const std::vector<cv::Point2f>& curvePoints, std::optional<float> yAxisMax = std::nullopt);
    cv::Mat plotPointsAndCurve(const std::string& chartTitle, const std::vector<float>& pointValues, const std::vector<cv::Point2f>& curvePoints, std::optional<float> yAxisMax = std::nullopt);
    cv::Mat plotTwoCurves(const std::string& chartTitle, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, std::optional<float> yAxisMax = std::nullopt);
    cv::Mat plotTwoPointsAndCurve(const std::string& chartTitle, std::vector<cv::Point2f>& discretePoints1, std::vector<cv::Point2f>& curvePoints1,
        std::vector<cv::Point2f>& discretePoints2, std::vector<cv::Point2f>& curvePoints2, std::optional<float> yAxisMax = std::nullopt);
        
    void tryCvPlot();
}
