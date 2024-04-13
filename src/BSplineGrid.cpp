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
}
