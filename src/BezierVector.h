#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// A Bezier curve/surface is by definition just a single "cell" or section of curve.
namespace CvImageDeform
{
    extern const float THIRDS[];

    cv::Point2f evalBezierCurveCubicPoint(const std::vector<cv::Point2f>& controlPoints, float t);
    cv::Point3f evalBezierSurfaceCubicPoint(const std::vector<std::vector<cv::Point3f>>& controlPoints, float u, float v);

    void evalBezierCurveCubic(const std::vector<cv::Point2f>& controlPoints, int nPoints, std::vector<cv::Point2f>& outputPoints);

    /**
    * @brief Evaluate a single Bezier patch using 4x4 control points.
    * @param controlPoints 4x4 control points.
    * @param nPointsDim Number of points to output, in each dimension, so nPointsDim^2 total points.
    * @param yStart The coordinate of the first control point.
    * @param outputPoints Output points, nPointsDim^2 in total.
    */
    void evalBezierSurfaceCubic(const std::vector<std::vector<cv::Point3f>>& controlPoints, int nPointsDim, int yStart, std::vector<std::vector<cv::Point3f>>& outputPoints);

    std::vector<cv::Point2f> fitBezierCurveCubic(const std::vector<cv::Point2f>& threePoints);
}
