#include "BezierVector.h"
#include "BSplineVector.h"
#include "BSplineMiscUtil.h"
#include "ImageUtil.h"

using namespace std;
using namespace CppOpenCVUtil;

namespace CvImageDeform
{
    // nPoints: Number of result points to produce, per segment.
    vector<cv::Point2f> evalBSplineCurveCubic(vector<cv::Point2f> allPoints, int nPoints)
    {
        vector<cv::Point2f> points;

        // sliding window of 4 control points
        int n = allPoints.size() - 3;

        for (int wi = 0; wi < n; wi++)
        {
            // Control points are interpolations on the segments.
            vector<cv::Point2f> bezierControlPoints;
            cv::Point2f p1 = interpolate(allPoints[wi + 0], allPoints[wi + 1], 2.0 / 3.0);
            cv::Point2f p2 = interpolate(allPoints[wi + 1], allPoints[wi + 2], 1.0 / 3.0);
            cv::Point2f p3 = interpolate(allPoints[wi + 1], allPoints[wi + 2], 2.0 / 3.0);
            cv::Point2f p4 = interpolate(allPoints[wi + 2], allPoints[wi + 3], 1.0 / 3.0);

            bezierControlPoints.push_back(interpolate(p1, p2, 0.5));
            bezierControlPoints.push_back(p2);
            bezierControlPoints.push_back(p3);
            bezierControlPoints.push_back(interpolate(p3, p4, 0.5));
            evalBezierCurveCubic(bezierControlPoints, nPoints, points);
        }

        return points;
    }

    void evalBSplineSurfaceCubic(const vector<vector<cv::Point3f>>& controlPoints, int nPointsDim, 
        vector<vector<cv::Point3f>>& outputPoints, bool doDebug)
    {
        // sliding window of 4x4 control points
        int nx = controlPoints[0].size();
        int ny = controlPoints.size();

        // for each 4x4 window
        for (int yi = 0; yi < ny - 3; yi++)
        {
            for (int xi = 0; xi < nx - 3; xi++)
            {
                // xi, yi defines the origin (indices) of a 4x4 window.

                // Interpolate across x first.
                vector<vector<cv::Point3f>> windowControlPointsX;

                for (int wyi = 0; wyi < 4; wyi++)
                {
                    int y = yi + wyi;
                    cv::Point3f p1 = interpolate(controlPoints[y][xi + 0], controlPoints[y][xi + 1], 2.0 / 3.0);
                    cv::Point3f p2 = interpolate(controlPoints[y][xi + 1], controlPoints[y][xi + 2], 1.0 / 3.0);
                    cv::Point3f p3 = interpolate(controlPoints[y][xi + 1], controlPoints[y][xi + 2], 2.0 / 3.0);
                    cv::Point3f p4 = interpolate(controlPoints[y][xi + 2], controlPoints[y][xi + 3], 1.0 / 3.0);

                    vector<cv::Point3f> rowPoints;
                    rowPoints.push_back(interpolate(p1, p2, 0.5));
                    rowPoints.push_back(p2);
                    rowPoints.push_back(p3);
                    rowPoints.push_back(interpolate(p3, p4, 0.5));
                    windowControlPointsX.push_back(rowPoints);
                }

                if (doDebug) dumpPoints("Window Points X", windowControlPointsX);

                // Interpolate in y dir (on the already x-interpolated points, just 4x4 at this point).
                vector<vector<cv::Point3f>> windowControlPoints;
                for (int wxi = 0; wxi < 4; wxi++) windowControlPoints.push_back(vector<cv::Point3f>());

                for (int wxi = 0; wxi < 4; wxi++)
                {
                    cv::Point3f p1 = interpolate(windowControlPointsX[0][wxi], windowControlPointsX[1][wxi], 2.0 / 3.0);
                    cv::Point3f p2 = interpolate(windowControlPointsX[1][wxi], windowControlPointsX[2][wxi], 1.0 / 3.0);
                    cv::Point3f p3 = interpolate(windowControlPointsX[1][wxi], windowControlPointsX[2][wxi], 2.0 / 3.0);
                    cv::Point3f p4 = interpolate(windowControlPointsX[2][wxi], windowControlPointsX[3][wxi], 1.0 / 3.0);

                    windowControlPoints[0].push_back(interpolate(p1, p2, 0.5));
                    windowControlPoints[1].push_back(p2);
                    windowControlPoints[2].push_back(p3);
                    windowControlPoints[3].push_back(interpolate(p3, p4, 0.5));
                }

                if (doDebug) dumpPoints("Window Interpolated Points", windowControlPoints);
                evalBezierSurfaceCubic(windowControlPoints, nPointsDim, yi * nPointsDim, outputPoints);
                if (doDebug) dumpPoints("Eval Points", outputPoints);
            }
        }
    }

    std::vector<cv::Point2f> fitBSplineCurveCubic(const std::vector<cv::Point2f>& inputPoints, int nth)
    {
        if (nth < 3)
        {
            throw std::runtime_error("fitBSplineCurveCubic: nth must be >= 3");
        }

        vector<cv::Point2f> resultPoints;

        for (int i = 0; i < inputPoints.size() - nth; i += nth - 1)
        {
            vector<cv::Point2f> thisPoints;

            for (int j = 0; j < nth; j++)
            {
                thisPoints.push_back(inputPoints[i + j]);
            }

            vector<cv::Point2f> fitted = fitBezierCurveCubic(thisPoints);

            for (int j = 0; j < fitted.size(); j++)
            {
                resultPoints.push_back(fitted[j]);
            }
        }

        return resultPoints;
    }
}
