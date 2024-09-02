#pragma once
#include <vector>
#include <opencv2/opencv.hpp>


namespace CvImageDeform
{
    /**
     * Eval a single point in a 3-d bezier volume.
     * The computation produces a 4-d point but this only returns the 4th dim value.
     *
     * @param controlPointsZs A 3-d 4x4x4 mat of "Z" values of the bezier control points.
     */
    float evalBezierVolumeCubicPointMat3d(const cv::Mat& controlPointsZs, float u, float v, float w);
}
