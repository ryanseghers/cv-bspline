#pragma once
#include <opencv2/opencv.hpp>


namespace CvImageDeform
{
    // Function to evaluate the cubic B-spline basis functions
    void computeCubicBSplineWeights(float t, float* weights);

    float evaluateBSpline1d(const cv::Mat& volume, float x);
    float evaluateBSpline2d(const cv::Mat& volume, float x, float y);
    float evaluateBSpline3d(const cv::Mat& volume, float x, float y, float z);
}
