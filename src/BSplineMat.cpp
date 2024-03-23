#include "BezierMat.h"
#include "BSplineMat.h"
#include "BSplineMiscUtil.h"
#include "ImageUtil.h"

using namespace std;
using namespace CppOpenCVUtil;

namespace CvImageDeform
{
    // WARN: No need to compute the x's and y's, so see the overload instead.
    void computeBezierControlPointsFull(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPoints)
    {
        // Number of bezier control points.
        int nbr = (bSplineControlPointZs.rows - 1) * 3;
        int nbc = (bSplineControlPointZs.cols - 1) * 3;
        bezierControlPoints = cv::Mat::zeros(nbr, nbc, CV_32FC3);

        // These are fractions of a cell, and right now we are using 1-dim cells.
        float oneThird = 1.0f / 3.0f;
        float twoThird = 2.0f / 3.0f;

        // Interpolate within rows (so in the X direction).
        // For each b-spline cell, including margin cells.
        for (int r = 0; r < bSplineControlPointZs.rows - 1; r++)
        {
            float y = r; // uniform grid

            for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
            {
                float x = c; // uniform grid
                float zThis = bSplineControlPointZs.at<float>(r, c);
                
                // By our definition of a cell there is always a next point.
                float zNext = bSplineControlPointZs.at<float>(r, c + 1);

                cv::Point3f p2 = cv::Point3f(x + oneThird, y, interpolate(zThis, zNext, oneThird));
                cv::Point3f p3 = cv::Point3f(x + twoThird, y, interpolate(zThis, zNext, twoThird));

                // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
                if (c > 0)
                {
                    float zPrior = bSplineControlPointZs.at<float>(r, c - 1);
                    cv::Point3f p1 = cv::Point3f(x - oneThird, y, interpolate(zPrior, zThis, twoThird));
                    cv::Point3f mid = interpolate(p1, p2, 0.5f);
                    bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 0) = mid;
                }

                bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 1) = p2;
                bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 2) = p3;
            }
        }

        // Interpolate within cols (so in the Y direction) on the points already interpolated in the x-dir.
        // For each b-spline cell, including margin cells.
        for (int r = 0; r < bSplineControlPointZs.rows - 1; r++)
        {
            float y = r; // uniform grid
            int bri = r * 3; // bezier row index

            for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
            {
                // Cells have 3 control points so have to interpolate each of them.
                for (int bi = 0; bi < 3; bi++)
                {
                    float x = c + bi * oneThird; // uniform grid
                    int bci = c * 3 + bi; // bezier col index

                    cv::Point3f bptThis = bezierControlPoints.at<cv::Point3f>(bri, bci);

                    // By our definition of a cell there is always a next point.
                    cv::Point3f bptNext = bezierControlPoints.at<cv::Point3f>(bri + 1, bci);

                    // By definition of a cell there is always a next point.
                    cv::Point3f p2 = cv::Point3f(x, y + oneThird, interpolate(bptThis.z, bptNext.z, oneThird));
                    cv::Point3f p3 = cv::Point3f(x, y + twoThird, interpolate(bptThis.z, bptNext.z, twoThird));

                    // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
                    if (r > 0)
                    {
                        cv::Point3f bptPrior = bezierControlPoints.at<cv::Point3f>(bri - 1, bci);
                        cv::Point3f p1 = cv::Point3f(x, y - oneThird, interpolate(bptPrior.z, bptThis.z, twoThird));
                        cv::Point3f mid = interpolate(p1, p2, 0.5f);
                        bezierControlPoints.at<cv::Point3f>(bri + 0, bci) = mid;
                    }

                    bezierControlPoints.at<cv::Point3f>(bri + 1, bci) = p2;
                    bezierControlPoints.at<cv::Point3f>(bri + 2, bci) = p3;
                }
            }
        }
    }

    void computeBezierControlPoints(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPointsZs, bool doDebug)
    {
        // Number of bezier control points.
        // (We include top/left edge points/cells just for simplicity in coding, but they are not computable and not used.)
        int nbr = bSplineControlPointZs.rows * 3;
        int nbc = bSplineControlPointZs.cols * 3;
        bezierControlPointsZs = cv::Mat::zeros(nbr, nbc, CV_32FC1);

        // These are fractions of a cell, and right now we are using 1-dim cells.
        float oneThird = 1.0f / 3.0f;
        float twoThird = 2.0f / 3.0f;

        // Interpolate within rows (so in the X direction).
        // For each b-spline cell, including margin cells.
        for (int r = 0; r < bSplineControlPointZs.rows; r++)
        {
            float y = r; // uniform grid

            for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
            {
                float x = c; // uniform grid
                float zThis = bSplineControlPointZs.at<float>(r, c);

                // By our definition of a cell there is always a next point.
                float zNext = bSplineControlPointZs.at<float>(r, c + 1);

                cv::Point3f p2 = cv::Point3f(x + oneThird, y, interpolate(zThis, zNext, oneThird));
                cv::Point3f p3 = cv::Point3f(x + twoThird, y, interpolate(zThis, zNext, twoThird));

                // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
                if (c > 0)
                {
                    float zPrior = bSplineControlPointZs.at<float>(r, c - 1);
                    cv::Point3f p1 = cv::Point3f(x - oneThird, y, interpolate(zPrior, zThis, twoThird));
                    cv::Point3f mid = interpolate(p1, p2, 0.5f);
                    bezierControlPointsZs.at<float>(r * 3, c * 3 + 0) = mid.z;
                }

                bezierControlPointsZs.at<float>(r * 3, c * 3 + 1) = p2.z;
                bezierControlPointsZs.at<float>(r * 3, c * 3 + 2) = p3.z;
            }
        }

        if (doDebug) dumpMat("bezierControlPointsZs - after X dir interpolation", bezierControlPointsZs);

        // Interpolate within cols (so in the Y direction) on the points already interpolated in the x-dir.
        // For each b-spline cell, including margin cells.
        // But now we are working in the bezier control points, so 3 points per cell.
        for (int r = 0; r < bSplineControlPointZs.rows - 1; r++)
        {
            float y = r; // uniform grid
            int bri = r * 3; // bezier row index

            for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
            {
                // Cells have 3 control points so have to interpolate each of them.
                for (int bi = 0; bi < 3; bi++)
                {
                    float x = c + bi * oneThird; // uniform grid
                    int bci = c * 3 + bi; // bezier col index

                    float bptThis = bezierControlPointsZs.at<float>(bri, bci);

                    // By our definition of a cell there is always a next point.
                    float bptNext = bezierControlPointsZs.at<float>(bri + 3, bci);

                    // By definition of a cell there is always a next point.
                    float p2 = interpolate(bptThis, bptNext, oneThird);
                    float p3 = interpolate(bptThis, bptNext, twoThird);

                    // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
                    if (r > 0)
                    {
                        float bptPrior = bezierControlPointsZs.at<float>(bri - 3, bci);
                        float p1 = interpolate(bptPrior, bptThis, twoThird);
                        float mid = interpolate(p1, p2, 0.5f);
                        bezierControlPointsZs.at<float>(bri + 0, bci) = mid;
                    }

                    bezierControlPointsZs.at<float>(bri + 1, bci) = p2;
                    bezierControlPointsZs.at<float>(bri + 2, bci) = p3;
                }
            }
        }
    }

    //// Stepping stone: Uses vector matrix.
    //void evalBSplineSurfaceCubic2(const cv::Mat& bezierControlPointsZs, int nPointsDim, std::vector<std::vector<cv::Point3f>>& outputPoints)
    //{
    //    // Iterate by cells.
    //    // Cannot eval edge/margin cells.
    //    int nr = bezierControlPointsZs.rows / 3;
    //    int nc = bezierControlPointsZs.cols / 3;

    //    if ((nr <= 0) || (nc <= 0))
    //    {
    //        throw std::invalid_argument("Cannot eval B-Spline surface with no non-edge cells.");
    //    }

    //    // TODO: don't actually want smaller output, just 0's on edges: mat will do this for me
    //    for (int r = 1; r < nr - 2; r++)
    //    {
    //        outputPoints.push_back(vector<cv::Point3f>());

    //        for (int c = 1; c < nc - 2; c++)
    //        {
    //            vector<vector<cv::Point3f>> windowControlPoints = matToPoints(bezierControlPointsZs(cv::Rect(c * 3, r * 3, 4, 4)));
    //            evalBezierSurfaceCubic(windowControlPoints, nPointsDim, (r - 1) * nPointsDim, outputPoints);
    //        }
    //    }
    //}

    void evalBSplineSurfaceCubicPrecomputedMat(const cv::Mat& bezierControlPointsZs, int nPointsDim, cv::Mat& outputMat)
    {
        // Iterate by cells.
        int nr = bezierControlPointsZs.rows / 3;
        int nc = bezierControlPointsZs.cols / 3;

        if ((nr <= 0) || (nc <= 0))
        {
            throw std::invalid_argument("Cannot eval B-Spline surface with no non-edge cells.");
        }

        ImageUtil::ensureMat(outputMat, nr * nPointsDim, nc * nPointsDim, CV_32FC3);

        // Decided not to clear edge cells here, for perf.

        // Eval non-edge cells.
        for (int r = 1; r < nr - 1; r++)
        {
            for (int c = 1; c < nc - 1; c++)
            {
                cv::Rect bezierCellRoi = cv::Rect(c * 3, r * 3, 4, 4);
                cv::Rect outputRoi = cv::Rect(c * nPointsDim, r * nPointsDim, nPointsDim, nPointsDim);
                cv::Mat outputMatRoi = outputMat(outputRoi);
                //evalBezierSurfaceCubicMat(bezierControlPointsZs(bezierCellRoi), nPointsDim, (float)c, (float)r, outputMatRoi);
                evalBezierSurfaceCubicMatAvx(bezierControlPointsZs(bezierCellRoi), nPointsDim, (float)c, (float)r, outputMatRoi);
            }
        }
    }

    void evalBSplineSurfaceCubicMat(const cv::Mat& controlPointZs, int nPointsDim, cv::Mat& outputMat)
    {
        cv::Mat bezierControlPointsZs;
        computeBezierControlPoints(controlPointZs, bezierControlPointsZs);
        evalBSplineSurfaceCubicPrecomputedMat(bezierControlPointsZs, nPointsDim, outputMat);
    }
}
