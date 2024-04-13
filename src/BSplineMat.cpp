#include "BezierMat.h"
#include "BSplineMat.h"
#include "BSplineMiscUtil.h"
#include "ImageUtil.h"

using namespace std;
using namespace CppOpenCVUtil;

namespace CvImageDeform
{
    // These are fractions of a cell, and right now we are using 1x1 cells.
    const float THIRDS[3] = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f };
    const float oneThird = 1.0f / 3.0f;
    const float twoThird = 2.0f / 3.0f;

    // WARN: No need to compute the x's and y's, so see the overload instead.
    void computeBezierControlPointsFull(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPoints)
    {
        //// Number of bezier control points.
        //int nbr = (bSplineControlPointZs.rows - 1) * 3;
        //int nbc = (bSplineControlPointZs.cols - 1) * 3;
        //bezierControlPoints = cv::Mat::zeros(nbr, nbc, CV_32FC3);

        //// These are fractions of a cell, and right now we are using 1-dim cells.
        //float oneThird = 1.0f / 3.0f;
        //float twoThird = 2.0f / 3.0f;

        //// Interpolate within rows (so in the X direction).
        //// For each b-spline cell, including margin cells.
        //for (int r = 0; r < bSplineControlPointZs.rows - 1; r++)
        //{
        //    float y = r; // uniform grid

        //    for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
        //    {
        //        float x = c; // uniform grid
        //        float zThis = bSplineControlPointZs.at<float>(r, c);
        //        
        //        // By our definition of a cell there is always a next point.
        //        float zNext = bSplineControlPointZs.at<float>(r, c + 1);

        //        cv::Point3f p2 = cv::Point3f(x + oneThird, y, interpolate(zThis, zNext, oneThird));
        //        cv::Point3f p3 = cv::Point3f(x + twoThird, y, interpolate(zThis, zNext, twoThird));

        //        // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
        //        if (c > 0)
        //        {
        //            float zPrior = bSplineControlPointZs.at<float>(r, c - 1);
        //            cv::Point3f p1 = cv::Point3f(x - oneThird, y, interpolate(zPrior, zThis, twoThird));
        //            cv::Point3f mid = interpolate(p1, p2, 0.5f);
        //            bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 0) = mid;
        //        }

        //        bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 1) = p2;
        //        bezierControlPoints.at<cv::Point3f>(r * 3, c * 3 + 2) = p3;
        //    }
        //}

        //// Interpolate within cols (so in the Y direction) on the points already interpolated in the x-dir.
        //// For each b-spline cell, including margin cells.
        //for (int r = 0; r < bSplineControlPointZs.rows - 1; r++)
        //{
        //    float y = r; // uniform grid
        //    int bri = r * 3; // bezier row index

        //    for (int c = 0; c < bSplineControlPointZs.cols - 1; c++)
        //    {
        //        // Cells have 3 control points so have to interpolate each of them.
        //        for (int bi = 0; bi < 3; bi++)
        //        {
        //            float x = c + bi * oneThird; // uniform grid
        //            int bci = c * 3 + bi; // bezier col index

        //            cv::Point3f bptThis = bezierControlPoints.at<cv::Point3f>(bri, bci);

        //            // By our definition of a cell there is always a next point.
        //            cv::Point3f bptNext = bezierControlPoints.at<cv::Point3f>(bri + 1, bci);

        //            // By definition of a cell there is always a next point.
        //            cv::Point3f p2 = cv::Point3f(x, y + oneThird, interpolate(bptThis.z, bptNext.z, oneThird));
        //            cv::Point3f p3 = cv::Point3f(x, y + twoThird, interpolate(bptThis.z, bptNext.z, twoThird));

        //            // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
        //            if (r > 0)
        //            {
        //                cv::Point3f bptPrior = bezierControlPoints.at<cv::Point3f>(bri - 1, bci);
        //                cv::Point3f p1 = cv::Point3f(x, y - oneThird, interpolate(bptPrior.z, bptThis.z, twoThird));
        //                cv::Point3f mid = interpolate(p1, p2, 0.5f);
        //                bezierControlPoints.at<cv::Point3f>(bri + 0, bci) = mid;
        //            }

        //            bezierControlPoints.at<cv::Point3f>(bri + 1, bci) = p2;
        //            bezierControlPoints.at<cv::Point3f>(bri + 2, bci) = p3;
        //        }
        //    }
        //}
    }

    /**
     * @brief 
     * This can be called for fist and last cell rows, but not for first and last cell cols.
     */
    void horizontalBSplineInterp(float* bsp, int bspStride, int r, int c, float* pOut)
    {
        float zThis = bsp[r * bspStride + c];
        float zNext = bsp[r * bspStride + c + 1]; // By our definition of a cell there is always a next point.

        pOut[1] = interpolate(zThis, zNext, oneThird);
        pOut[2] = interpolate(zThis, zNext, twoThird);

        float zPrior = bsp[r * bspStride + c - 1];
        float p1 = interpolate(zPrior, zThis, twoThird);
        float mid = interpolate(p1, pOut[1], 0.5f);
        pOut[0] = mid;
    }

    /**
     * @brief Both directions, just one cell, not referencing other cells' results.
     * This cannot be called on edge cells.
     */
    void computeBezierControlPointsSingleCellIsolated(const cv::Mat& bSplineControlPointZs, int r, int c, cv::Mat& bezierControlPointsZs)
    {
        // Setup for indexing the matrices.
        int bri = r * 3; // bezier row index
        int bci = c * 3; // bezier col index

        float* bsp = (float*)bSplineControlPointZs.data;
        int bspStride = bSplineControlPointZs.step[0] / bSplineControlPointZs.elemSize();

        float* bzp = (float*)bezierControlPointsZs.data;
        int bzpStride = bezierControlPointsZs.step[0] / bezierControlPointsZs.elemSize();

        // Do horizontal into 3 tmp rows.
        // (only using 3 cols but pad for size)
        float bzPriorRow[4] = { 0.0f };
        float bzThisRow[4] = { 0.0f };
        float bzNextRow[4] = { 0.0f };

        horizontalBSplineInterp(bsp, bspStride, r - 1, c, &bzPriorRow[0]);
        horizontalBSplineInterp(bsp, bspStride, r, c, &bzThisRow[0]);
        horizontalBSplineInterp(bsp, bspStride, r + 1, c, &bzNextRow[0]);

        // Vertical: same interp pattern but on the horizontal results.
        for (int bi = 0; bi < 3; bi++) // bezier col
        {
            float bptPrior = bzPriorRow[bi];
            float bptThis = bzThisRow[bi];
            float bptNext = bzNextRow[bi];

            float p1 = interpolate(bptPrior, bptThis, twoThird);
            float p2 = interpolate(bptThis, bptNext, oneThird);
            float p3 = interpolate(bptThis, bptNext, twoThird);

            float mid = interpolate(p1, p2, 0.5f);

            bzp[bzpStride * (bri + 0) + bci + bi] = mid;

            bzp[bzpStride * (bri + 1) + bci + bi] = p2;
            bzp[bzpStride * (bri + 2) + bci + bi] = p3;
        }
    }

    /**
    * @brief Compute/interpolate the 3 bezier control points for a single row of a single cell.
    * This can be called for any cell including edges.
    * @param bsp Pointer to start of b-spline control points matrix.
    * @param bspStride Stride (in elements) of b-spline control points matrix.
    * @param r Row index of cell.
    * @param c Column index of cell.
    * @param numCols Number of columns in b-spline control points matrix.
    * @param pOut Pointer to start of output array.
    */
    void horizontalBSplineInterpFull(float* bsp, int bspStride, int r, int c, int numCols, float* pOut)
    {
        float zThis = bsp[r * bspStride + c];

        if (c == 0)
        {
            // first cell col
            pOut[0] = zThis;

            float zNext = bsp[r * bspStride + c + 1];
            pOut[1] = interpolate(zThis, zNext, oneThird);
            pOut[2] = interpolate(zThis, zNext, twoThird);
        }
        else if (c == numCols - 1)
        {
            // last cell col
            pOut[0] = zThis;
            pOut[1] = 0.0f;
            pOut[2] = 0.0f;
        }
        else
        {
            // middle cell col
            float zPrior = bsp[r * bspStride + c - 1];
            float zNext = bsp[r * bspStride + c + 1];
            float p1 = interpolate(zPrior, zThis, twoThird);
            float p2 = interpolate(zThis, zNext, oneThird);
            float p3 = interpolate(zThis, zNext, twoThird);

            float mid = interpolate(p1, p2, 0.5f);
            pOut[0] = mid;
            pOut[1] = p2;
            pOut[2] = p3;
        }
    }

    /**
    * @brief Both directions, just one cell, not referencing other cells' results.
    * This can be called on edge cells.
    */
    void computeBezierControlPointsSingleCellIsolatedFull(const cv::Mat& bSplineControlPointZs, int r, int c, cv::Mat& bezierControlPointsZs)
    {
        // Setup for indexing the matrices.
        int bri = r * 3; // bezier row index
        int bci = c * 3; // bezier col index

        int rMax = bSplineControlPointZs.rows - 1;

        float* bsp = (float*)bSplineControlPointZs.data;
        int bspNumCols = bSplineControlPointZs.cols;
        int bspStride = bSplineControlPointZs.step[0] / bSplineControlPointZs.elemSize();

        float* bzp = (float*)bezierControlPointsZs.data;
        int bzpStride = bezierControlPointsZs.step[0] / bezierControlPointsZs.elemSize();

        // Do horizontal into 3 tmp rows.
        // (only using 3 cols but pad for size)
        float bzPriorRow[4] = { 0.0f };
        float bzThisRow[4] = { 0.0f };
        float bzNextRow[4] = { 0.0f };

        if (r > 0)
        {
            horizontalBSplineInterpFull(bsp, bspStride, r - 1, c, bspNumCols, &bzPriorRow[0]);
        }

        horizontalBSplineInterpFull(bsp, bspStride, r, c, bspNumCols, &bzThisRow[0]);

        if (r < rMax)
        {
            horizontalBSplineInterpFull(bsp, bspStride, r + 1, c, bspNumCols, &bzNextRow[0]);
        }

        // don't do last two cols of last cell column
        int biMax = (c == bSplineControlPointZs.cols - 1) ? 1 : 3;

        // Vertical
        for (int bi = 0; bi < biMax; bi++) // bezier col
        {
            float bptThis = bzThisRow[bi];

            if (r == 0)
            {
                // first row
                bzp[bzpStride * (bri + 0) + bci + bi] = bptThis;

                float bptNext = bzNextRow[bi];
                float p2 = interpolate(bptThis, bptNext, oneThird);
                bzp[bzpStride * (bri + 1) + bci + bi] = p2;

                float p3 = interpolate(bptThis, bptNext, twoThird);
                bzp[bzpStride * (bri + 2) + bci + bi] = p3;
            }
            else if (r == rMax)
            {
                // last row
                bzp[bzpStride * (bri + 0) + bci + bi] = bptThis;
            }
            else
            {
                // middle rows
                float bptPrior = bzPriorRow[bi];
                float bptNext = bzNextRow[bi];

                float p1 = interpolate(bptPrior, bptThis, twoThird);
                float p2 = interpolate(bptThis, bptNext, oneThird);
                float p3 = interpolate(bptThis, bptNext, twoThird);

                float mid = interpolate(p1, p2, 0.5f);
                bzp[bzpStride * (bri + 0) + bci + bi] = mid;
                bzp[bzpStride * (bri + 1) + bci + bi] = p2;
                bzp[bzpStride * (bri + 2) + bci + bi] = p3;
            }
        }
    }

    void computeBezierControlPointsIsolated(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPointsZs, bool doFull)
    {
        // Number of bezier control points.
        // See the readme for the reasoning behind these sizes.
        int nbr = bSplineControlPointZs.rows * 3;
        int nbc = bSplineControlPointZs.cols * 3;
        bezierControlPointsZs = cv::Mat::zeros(nbr, nbc, CV_32FC1);

        // For each b-spline cell.
        if (doFull)
        {
            for (int r = 0; r < bSplineControlPointZs.rows; r++)
            {
                for (int c = 0; c < bSplineControlPointZs.cols; c++)
                {
                    computeBezierControlPointsSingleCellIsolatedFull(bSplineControlPointZs, r, c, bezierControlPointsZs);
                }
            }
        }
        else
        {
            assert(false); // not finished: only Full computes the first row/col of edge cells and that is required

            for (int r = 1; r < bSplineControlPointZs.rows - 1; r++)
            {
                for (int c = 1; c < bSplineControlPointZs.cols - 1; c++)
                {
                    computeBezierControlPointsSingleCellIsolated(bSplineControlPointZs, r, c, bezierControlPointsZs);
                }
            }

            //// Last row
            //// There is no next row so cannot do vertical interpolation.
            //for (int c = 1; c < bSplineControlPointZs.cols - 1; c++)
            //{
            //    computeBezierControlPointsSingleCellIsolated(bSplineControlPointZs, r, c, bezierControlPointsZs);
            //}
        }
    }

    void horzBSplineInterp(const cv::Mat& mat, cv::Mat& outMat)
    {
        int nbr = mat.rows;
        int nbc = mat.cols * 3;
        outMat = cv::Mat::zeros(nbr, nbc, CV_32FC1);

        for (int r = 0; r < nbr; r++)
        {
            for (int c = 0; c < mat.cols; c++)
            {
                if (c == 0)
                {
                    float zThis = mat.at<float>(r, c);
                    float zNext = mat.at<float>(r, c + 1);

                    float p2 = interpolate(zThis, zNext, oneThird);
                    float p3 = interpolate(zThis, zNext, twoThird);

                    outMat.at<float>(r, c * 3 + 0) = zThis;
                    outMat.at<float>(r, c * 3 + 1) = p2;
                    outMat.at<float>(r, c * 3 + 2) = p3;
                }
                else if (c == mat.cols - 1)
                {
                    float zThis = mat.at<float>(r, c);
                    outMat.at<float>(r, c * 3 + 0) = zThis;
                }
                else
                {
                    float zPrior = mat.at<float>(r, c - 1);
                    float zThis = mat.at<float>(r, c);
                    float zNext = mat.at<float>(r, c + 1);

                    float p2 = interpolate(zThis, zNext, oneThird);
                    float p1 = interpolate(zPrior, zThis, twoThird);
                    float mid = interpolate(p1, p2, 0.5f);
                    float p3 = interpolate(zThis, zNext, twoThird);

                    outMat.at<float>(r, c * 3 + 0) = mid;
                    outMat.at<float>(r, c * 3 + 1) = p2;
                    outMat.at<float>(r, c * 3 + 2) = p3;
                }
            }
        }
    }

    void computeBezierControlPointsSimple(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPointsZs)
    {
        cv::Mat tmp1, tmp2;
        horzBSplineInterp(bSplineControlPointZs, tmp1);
        cv::transpose(tmp1, tmp2);
        horzBSplineInterp(tmp2, tmp1);
        cv::transpose(tmp1, bezierControlPointsZs);
    }

    // doFull is not finished, relatively tricky to implement.
    void computeBezierControlPoints(const cv::Mat& bSplineControlPointZs, cv::Mat& bezierControlPointsZs, bool doFull)
    {
        assert(doFull == false); // not implemented yet

        // Number of bezier control points.
        // See the readme for the reasoning behind these sizes.
        int nbr = bSplineControlPointZs.rows * 3;
        int nbc = bSplineControlPointZs.cols * 3;
        bezierControlPointsZs = cv::Mat::zeros(nbr, nbc, CV_32FC1);

        int rStart = doFull ? 0 : 1;
        int cStart = doFull ? 0 : 1;

        // We need a temp matrix for the horizontal interpolation results because otherwise some
        // get overwritten before they are used in the vertical interpolation.
        cv::Mat horzZs = cv::Mat::zeros(nbr + 3, nbc + 3, CV_32FC1);

        // Horizontal interpolation
        // For all rows because they are used for vertical.
        for (int r = 0; r < bSplineControlPointZs.rows; r++)
        {
            for (int c = cStart; c < bSplineControlPointZs.cols; c++)
            {
                float zThis = bSplineControlPointZs.at<float>(r, c);

                if (c == 0)
                {
                    float zNext = bSplineControlPointZs.at<float>(r, c + 1);
                    float p2 = interpolate(zThis, zNext, oneThird);
                    float p3 = interpolate(zThis, zNext, twoThird);
                    horzZs.at<float>(r * 3, c * 3 + 0) = zThis;
                    horzZs.at<float>(r * 3, c * 3 + 1) = p2;
                    horzZs.at<float>(r * 3, c * 3 + 2) = p3;
                }
                else if (c == bSplineControlPointZs.cols - 1)
                {
                    horzZs.at<float>(r * 3, c * 3 + 0) = zThis;
                }
                else
                {
                    float zPrior = (c > 0) ? bSplineControlPointZs.at<float>(r, c - 1) : zThis; // for full
                    float zNext = bSplineControlPointZs.at<float>(r, c + 1);

                    float p2 = interpolate(zThis, zNext, oneThird);
                    float p3 = interpolate(zThis, zNext, twoThird);
                    float p1 = interpolate(zPrior, zThis, twoThird);
                    float mid = interpolate(p1, p2, 0.5f);

                    horzZs.at<float>(r * 3, c * 3 + 0) = mid;
                    horzZs.at<float>(r * 3, c * 3 + 1) = p2;
                    horzZs.at<float>(r * 3, c * 3 + 2) = p3;
                }
            }

            //// For full write the last point.
            //int c = bSplineControlPointZs.cols - 1;

            //for (int ri = 0; ri < 3; ri++)
            //{
            //    horzZs.at<float>(r * 3 + ri, c * 3) = bSplineControlPointZs.at<float>(r, c);
            //}
        }

        //dumpMat("bezierControlPointsZs - after horizontal interpolation", horzZs);
        //return;

        // Vertical interpolation on the points already interpolated horizontally.
        // For each computable b-spline cell.
        // Now we are working in the bezier control points, so 3 points per cell.
        for (int r = rStart; r < bSplineControlPointZs.rows - 1; r++)
        {
            int bri = r * 3; // bezier row index

            for (int c = 0; c < bSplineControlPointZs.cols; c++)
            {
                // Cells have 3 control points so have to interpolate each of them.
                for (int bi = 0; bi < 3; bi++)
                {
                    int bci = c * 3 + bi; // bezier col index

                    // these refer to horizontally interpolated points
                    float bptThis = horzZs.at<float>(bri, bci);
                    float bptPrior = (bri > 2) ? horzZs.at<float>(bri - 3, bci) : bptThis;
                    float bptNext = horzZs.at<float>(bri + 3, bci);

                    float p1 = interpolate(bptPrior, bptThis, twoThird);
                    float p2 = interpolate(bptThis, bptNext, oneThird);
                    float p3 = interpolate(bptThis, bptNext, twoThird);

                    float mid = interpolate(p1, p2, 0.5f);

                    bezierControlPointsZs.at<float>(bri + 0, bci) = mid;
                    bezierControlPointsZs.at<float>(bri + 1, bci) = p2;
                    bezierControlPointsZs.at<float>(bri + 2, bci) = p3;
                }
            }
        }

        // For full write the last row.
        int r = bSplineControlPointZs.rows - 1;
        int bri = r * 3;

        for (int c = 0; c < bezierControlPointsZs.cols; c++)
        {
            bezierControlPointsZs.at<float>(bri, c) = horzZs.at<float>(bri, c);
        }
    }

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

        // We can't eval last row/col of cells, those only have the first coeff computed.
        for (int r = 0; r < nr - 1; r++)
        {
            for (int c = 0; c < nc - 1; c++)
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
        computeBezierControlPoints(controlPointZs, bezierControlPointsZs, false);
        evalBSplineSurfaceCubicPrecomputedMat(bezierControlPointsZs, nPointsDim, outputMat);
    }
}
