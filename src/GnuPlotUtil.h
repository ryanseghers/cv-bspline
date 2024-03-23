#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    void gnuPlot3dSurface(const std::string& chartTitle, const std::vector<std::vector<cv::Point3f>>& points);

    /**
     * @brief Plot a uniform grid of Z values.
    */
    void gnuPlot3dSurfaceZs(const std::string& chartTitle, const cv::Mat& matZs);
    void gnuPlot3dSurfaceMat(const std::string& chartTitle, const cv::Mat& m);
    
    void tryGnuPlot();
}
