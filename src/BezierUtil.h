#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    extern const float THIRDS[];

    /**
    * @brief Linear interpolation between two values or points.
    */
    template <typename T>
    T interpolate(const T p0, const T p1, float t)
    {
        return p0 + (p1 - p0) * t;
    }

    /**
     * @brief Cubic bezier polynomial term.
     */
    float bezierPolyTerm(int i, float t);
}
