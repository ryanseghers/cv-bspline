#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    /**
     * @brief Represents a B-spline grid.
     * 
     * See BSpline.h for discussion of numbers of B-Spline and Bezier control points.
     * 
     * The roi where the surface is defined (computable) is called the "surface roi".
     * Each grid cell in the surface roi is a patch.
     * Each patch is evaluated as a bezier surface.
    */
    class BSplineGrid
    {
    private:
        // Size of the surface roi, not the number of control points.
        int _rows = 0;

        // Size of the surface roi, not the number of control points.
        int _cols = 0;

        // This class can shift and scale with origin and scale.
        // This is a real-world origin of the surface roi.
        cv::Point2f _origin;

        // Multiple the control points by this to get the real-world position.
        float _scale = 1.0f;

        // The x and y are computable, so only need to store Z's.
        // This is larger than the surface roi, so that the surface roi can be computed.
        cv::Mat _controlPointZs;

    public:
        BSplineGrid(int rows, int cols);
        BSplineGrid(int rows, int cols, cv::Point2f origin, float scale);

        int rows() const { return _rows; }
        int cols() const { return _cols; }
        cv::Point2f origin() const { return _origin; }
        float scale() const { return _scale; }

        /**
        * @brief This doesn't clone so you get the original data.
        */
        cv::Mat getControlPointZs() const 
        { 
            return _controlPointZs; 
        }

        void setControlPointZs(cv::Mat& m) 
        { 
            _controlPointZs = m; 
        }

        void fillRandomControlPoints(float min, float max);

        /**
        * @brief Evaluate the b-spline surface over points in the surface roi, with nPoints x nPoints per cell.
        * This is usually just for rendering, so you eval many more points on the surface to fill in the rendered surface.
        * Returns a 1-channel matrix with the z coordinates of the sample points.
        * 
        * @param nPointsDim Per-dimension mumber of points to evaluate per cell, so nDimPerCell * nDimPerCell total points.
        */
        cv::Mat evalSurface(int nPointsDim);

        cv::Point2i getControlPointIndices(int row, int col) const
        {
            return cv::Point2i(col + 1, row + 1);
        }
    };
}
