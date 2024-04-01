#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    /**
    * @brief Linear interpolation between two values or points.
    */
    template <typename T>
    T interpolate(const T p0, const T p1, float t)
    {
        return p0 + (p1 - p0) * t;
    }

    void dumpPoints(const char* title, const std::vector<std::vector<cv::Point3f>>& points);
    void dumpMat(const char* title, const cv::Mat& m, int cellWidth = 0, int cellHeight = 0);
    void printDiffMat(const cv::Mat& mat1, const cv::Mat& mat2, int cellWidth = 0, int cellHeight = 0);

    /**
    * @brief Convert a mat of Z values to a vector of 3D points.
    * @param mat One-channel float z values.
    * @param origin Origin for x and y values (not z).
    * @param xyScale Scale for x and y values (not z).
    */
    std::vector<std::vector<cv::Point3f>> matToPoints(const cv::Mat& mat, cv::Point2f origin = cv::Point2f(), float xyScale = 1.0f);

    /**
     * @brief Convert a vector of 3D points to a one-channel float mat of Z values.
    */
    cv::Mat pointsToMatZs(const std::vector<std::vector<cv::Point3f>>& points);

    void setDummyValues(cv::Mat& m);
}
