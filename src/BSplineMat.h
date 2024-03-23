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
    /**
     * @brief For a uniform grid of B-Spline control points, compute (interpolate) the corresponding Bezier control points.
     * 
     * The output matrix has three points per input point, including edge cells despite them not being computable.
     * The output matrix has three points per cell, including edge cells despite them not being computable
     * because the first cell cannot have a first point and last cell computation needs first point of next cell.
     * 
     * @param bSplineControlPointZs Just Z values
     * @param bezierControlPoints Just Z values.
    */
    void computeBezierControlPoints(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPointsZs, bool doDebug = false);

    /**
    * @brief Eval a B-Spline surface using the given (pre-computed) Bezier control points.
    * This doesn't zero the edge cells of the output matrix.
    * 
    * @param bezierControlPointsZs Just Z values. Pre-computed, for example by computeBezierControlPoints.
    * @param nPointsDim Number of points per cell to sample.
    * @param outputMat Output mat of cv::Point3f. The edge cells are not zeroed.
    */
    void evalBSplineSurfaceCubicPrecomputedMat(const cv::Mat& bezierControlPointsZs, int nPointsDim, cv::Mat& outputMat);

    /**
     * @brief Eval a B-Spline surface using the B-spline control points.
    */
    void evalBSplineSurfaceCubicMat(const cv::Mat& controlPointZs, int nPointsDim, cv::Mat& outputMat);
}
