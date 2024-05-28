#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// This is for Uniform B-spline, meaning control points are on a uniform grid, evenly spaced in x and y.
// In the uniform case, the Bezier control points are also on a uniform grid, because they are just on thirds of the B-spline uniform grid.
// 
// A segment/cell requires a last edge point, so in other words the last control point ends a segment/cell and does not start/define a new one.
// In general this API doesn't hide the margin segments/cells, so caller has to know that the b-spline is
// not computable in those segments/cells.
// This uses the convention (talking in 1-d) that each segment has 3 bezier control points, the first one is the midpoint
// using a point from the previous segment/cell, and then the next two are the trivial thirds points.

namespace CvImageDeform
{
    std::vector<cv::Point2f> evalBSplineCurveCubic(std::vector<cv::Point2f> allPoints, int nPoints);

    /**
     * @brief Given a 2D array of uniform control points, evaluate a B-Spline surface.
     * @param controlPoints B-spline control points, just Z values since uniform grid and Z doesn't changed based on (uniform) x,y scaling.
     * @param nPointsDim per computable (non-margin) cell, in each dimension, so total output points count is nCells*nPointsDim^2
     * @param outputPoints 
    */
    void evalBSplineSurfaceCubic(const std::vector<std::vector<cv::Point3f>>& controlPoints, int nPointsDim, 
        std::vector<std::vector<cv::Point3f>>& outputPoints, bool doDebug = false);

    /**
    * @brief Fit the input (uniform, sorted) points with a B-spline curve.
    * @param inputPoints Three or more input points.
    * @param nth How many points define each B-spline cell. This must be gte 3.
    * @return The four Bezier control points.
    */
    std::vector<cv::Point2f> fitBSplineCurveCubic(const std::vector<cv::Point2f>& inputPoints, int nth);
}
