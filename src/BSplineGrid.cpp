#include "BSplineGrid.h"
#include "BSplineMat.h"
#include "BSplineMiscUtil.h"

using namespace std;

namespace CvImageDeform
{
    BSplineGrid::BSplineGrid(int rows, int cols)
    {
        this->_rows = rows;
        this->_cols = cols;

        // The computable area is smaller than the control points area.
        this->_controlPointZs = cv::Mat::zeros(rows + 3, cols + 3, CV_32FC1);
    }

    BSplineGrid::BSplineGrid(int rows, int cols, cv::Point2f origin, float scale)
    {
        this->_rows = rows;
        this->_cols = cols;
    }

    void BSplineGrid::fillRandomControlPoints(float min, float max)
    {
        cv::randu(this->_controlPointZs, min, max);
    }

    cv::Mat BSplineGrid::evalSurface(int nPointsDim)
    {
        cv::Mat evalMat;
        evalBSplineSurfaceCubicMat(_controlPointZs, nPointsDim, evalMat);
        cv::Rect roi(1 * nPointsDim, 1 * nPointsDim, _cols * nPointsDim, _rows * nPointsDim);
        return evalMat(roi);
    }

    float BSplineGrid::getZValue(cv::Point2i pt)
    {
        return _controlPointZs.at<float>(pt.y + 1, pt.x + 1);
    }

    float BSplineGrid::getZValue(int x, int y)
    {
        return _controlPointZs.at<float>(y + 1, x + 1);
    }

    cv::Mat BSplineGrid::renderField(int nPointsDim)
    {
        int width = _cols * nPointsDim;
        int height = _rows * nPointsDim;
        const int vectorDensity = 2;

        cv::Scalar circleColor(255, 255, 255);
        cv::Scalar lineColor(255, 255, 0);
        cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);

        for (int yi = 0; yi < _rows; yi++)
        {
            for (int xi = 0; xi < _cols; xi++)
            {
                cv::Point2i pt(xi * nPointsDim, yi * nPointsDim);
                cv::circle(img, pt, 7, circleColor);

                // vector (regardless of actual direction)
                // don't show them all else they overlap
                if ((yi % vectorDensity == 0) && (xi % vectorDensity == 0))
                {
                    float z = getZValue(xi, yi);
                    // scaling
                    z /= 2;
                    cv::Point2i pt2(pt.x + z, pt.y + z);
                    cv::line(img, pt, pt2, lineColor);
                }
            }
        }

        return img;
    }
}
