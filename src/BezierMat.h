#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// A Bezier curve/surface is by definition just a single "cell" or section of curve.
namespace CvImageDeform
{
    cv::Point3f evalBezierSurfaceCubicPointMat(const cv::Mat& controlPointsZs, float u, float v);
    cv::Point3f evalBezierSurfaceCubicPointMatSimd(const cv::Mat& controlPointsZs, float u, float v);
    cv::Point3f evalBezierSurfaceCubicPointMatSimd2(const cv::Mat& controlPointsZs, float u, float v);
    cv::Point3f evalBezierSurfaceCubicPointMatSimd3(const cv::Mat& controlPointsZs, float u, float v);

    /**
     * @brief Evaluate a single Bezier patch using 4x4 control points.
     * @param controlPointZs 
     * @param nPointsDim Number of points to output, in each dimension, so nPointsDim^2 total points.
     * @param xOrigin The coordinate of the first control point.
     * @param yOrigin The coordinate of the first control point.
     * @param outputMat cv::Point3f matrix to write to.
    */
    void evalBezierSurfaceCubicMat(const cv::Mat& controlPointZs, int nPointsDim, float xOrigin, float yOrigin, cv::Mat& outputMat);
    void evalBezierSurfaceCubicMatAvx(const cv::Mat& controlPointsZs, int nPointsDim, float xOrigin, float yOrigin, cv::Mat& outputMat);
}
