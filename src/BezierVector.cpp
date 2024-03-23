#include <immintrin.h>
#include "BezierVector.h"
#include "BezierUtil.h"

using namespace std;

namespace CvImageDeform
{
    // Single value of t.
    // Uses 4 control points.
    // t: from 0 to 1
    cv::Point2f evalBezierCurveCubicPoint(const std::vector<cv::Point2f>& controlPoints, float t)
    {
        float u = 1 - t;
        float b0 = u * u * u;
        float b1 = 3 * u * u * t;
        float b2 = 3 * u * t * t;
        float b3 = t * t * t;
        cv::Point2f p = b0 * controlPoints[0] + b1 * controlPoints[1] + b2 * controlPoints[2] + b3 * controlPoints[3];
        return p;
    }

    // From t = 0 to 1 via nPoints.
    // outputPoints: push_back()s the points
    void evalBezierCurveCubic(const std::vector<cv::Point2f>& controlPoints, int nPoints, std::vector<cv::Point2f>& outputPoints)
    {
        for (int i = 0; i < nPoints; i++)
        {
            float t = (float)i / (nPoints - 1);
            cv::Point2f p = evalBezierCurveCubicPoint(controlPoints, t);
            outputPoints.push_back(p);
        }
    }

    // Eval the bezier surface at a single point.
    // controlPoints is uniform 4x4 matrix
    // u,v are [0,1]
    cv::Point3f evalBezierSurfaceCubicPoint(const vector<vector<cv::Point3f>>& controlPoints, float u, float v) 
    {
        cv::Point3f point(0, 0, 0);

        for (int i = 0; i < 4; ++i) 
        {
            for (int j = 0; j < 4; ++j) 
            {
                point += bezierPolyTerm(i, u) * bezierPolyTerm(j, v) * controlPoints[i][j];
            }
        }

        return point;
    }

    cv::Point3f evalBezierSurfaceCubicPointUnrolled(const vector<vector<cv::Point3f>>& controlPoints, float u, float v) 
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float u1 = 1.0f - u;
        float v1 = 1.0f - v;
        float v13 = v1 * v1 * v1;
        float v12 = 3 * v * v1 * v1;
        float v21 = 3 * v * v * v1;
        float v3 = v * v * v;

        for (int i = 0; i < 4; ++i) 
        {
            float a = bezierPolyTerm(i, u) * v13;
            x += a * controlPoints[i][0].x;
            y += a * controlPoints[i][0].y;
            z += a * controlPoints[i][0].z;

            a = bezierPolyTerm(i, u) * v12;
            x += a * controlPoints[i][1].x;
            y += a * controlPoints[i][1].y;
            z += a * controlPoints[i][1].z;

            a = bezierPolyTerm(i, u) * v21;
            x += a * controlPoints[i][2].x;
            y += a * controlPoints[i][2].y;
            z += a * controlPoints[i][2].z;

            a = bezierPolyTerm(i, u) * v3;
            x += a * controlPoints[i][3].x;
            y += a * controlPoints[i][3].y;
            z += a * controlPoints[i][3].z;
        }

        //// See how it looks to leave out x,y interpolation
        //x = controlPoints[1][1].x() + v;
        //y = controlPoints[1][1].y() + u;

        return cv::Point3f(x, y, z);
    }

    // nPoints: each dimension, so total output points length nPoints^2
    // yStart: Index to start putting results into outputPoints. This will enlarge the outer (Y) vector if needed.
    // outputPoints: row/y-major matrix
    void evalBezierSurfaceCubic(const vector<vector<cv::Point3f>>& controlPoints, int nPointsDim, int yStart, vector<vector<cv::Point3f>>& outputPoints)
    {
        assert(controlPoints.size() == 4);
        assert(controlPoints[0].size() == 4);

        for (int yi = 0; yi < nPointsDim; yi++)
        {
            float v = (float)yi / nPointsDim;
            if (yStart + yi >= outputPoints.size()) outputPoints.push_back(vector<cv::Point3f>());

            for (int xi = 0; xi < nPointsDim; xi++)
            {
                float u = (float)xi / nPointsDim;
                cv::Point3f p = evalBezierSurfaceCubicPointUnrolled(controlPoints, v, u);
                outputPoints[yStart + yi].push_back(p);
            }
        }
    }
}
