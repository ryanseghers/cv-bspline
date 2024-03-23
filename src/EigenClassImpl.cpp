// This was my original main.cpp where I used Eigen classes everywhere.
// Then I decided to split this out into separate files, and also change to use cv types.
#include <string>
#include <chrono>
#include <ctime>
#include <future>
#include <optional>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <CvPlot/cvplot.h>

using Eigen::Matrix2f;
using Eigen::Vector2f;
using Eigen::Vector3f;

#include "C:/Dev/gnuplot-iostream/gnuplot-iostream.h"

#include "ImageUtil.h"

using namespace std;
using namespace std::chrono;
using namespace CppOpenCVUtil;

void splitPoints(const vector<Vector2f>& points, vector<float>& xs, vector<float>& ys)
{
    for (const auto& p : points)
    {
        xs.push_back(p.x());
        ys.push_back(p.y());
    }
}

void cvPlotAddPointSeries(CvPlot::Axes& axes, const vector<Vector2f>& points, bool showPoints)
{
    vector<float> xs, ys;
    splitPoints(points, xs, ys);
    CvPlot::Series& series = axes.create<CvPlot::Series>(xs, ys, showPoints ? "-r" : "-b");

    if (showPoints)
    {
        series.setMarkerType(CvPlot::MarkerType::Circle);
        series.setMarkerSize(16);
    }

    series.setLineType(CvPlot::LineType::Solid);
    series.setLineWidth(1);
}

CvPlot::Axes makeScatterPlotAxes(const string& chartTitle)
{
    auto axes = CvPlot::makePlotAxes();
    axes.title(chartTitle);

    axes.setYTight(false);
    axes.setXTight(false);
    axes.xLabel("X");
    axes.yLabel("Y");
    return axes;
}

cv::Mat renderPlot(const string& chartTitle, const vector<Vector2f>& points, const vector<Vector2f>& curvePoints)
{
    int drawHeight = 1024;
    int drawWidth = 1024;

    // the hist image to render to and then display
    cv::Mat img;
    img.create(drawHeight, drawWidth, CV_8UC3);

    auto axes = makeScatterPlotAxes(chartTitle);

    if (!points.empty())
    {
        cvPlotAddPointSeries(axes, points, true);
    }

    if (!curvePoints.empty())
    {
        cvPlotAddPointSeries(axes, curvePoints, false);
    }

    img = axes.render(drawHeight, drawWidth);
    return img;
}

cv::Mat plotPoints(const string& chartTitle, const vector<Vector2f>& points)
{
    vector<Vector2f> empty;
    return renderPlot(chartTitle, points, empty);
}

cv::Mat plotPointsAndCurve(const string& chartTitle, const vector<Vector2f>& points, const vector<Vector2f>& curvePoints)
{
    return renderPlot(chartTitle, points, curvePoints);
}

// Single value of t.
// Uses 4 control points.
// t: from 0 to 1
Vector2f evalBezierCurveCubic(const vector<Vector2f>& controlPoints, float t)
{
    float u = 1 - t;
    float b0 = u * u * u;
    float b1 = 3 * u * u * t;
    float b2 = 3 * u * t * t;
    float b3 = t * t * t;
    Vector2f p = b0 * controlPoints[0] + b1 * controlPoints[1] + b2 * controlPoints[2] + b3 * controlPoints[3];
    return p;
}

// From t = 0 to 1 via nPoints.
// outputPoints: push_back()s the points
void evalBezierCurveCubic(const vector<Vector2f>& controlPoints, int nPoints, vector<Vector2f>& outputPoints)
{
    for (int i = 0; i < nPoints; i++)
    {
        float t = (float)i / (nPoints - 1);
        Vector2f p = evalBezierCurveCubic(controlPoints, t);
        outputPoints.push_back(p);
    }
}

Vector2f interpolate(const Vector2f& p0, const Vector2f& p1, float t)
{
    return p0 + (p1 - p0) * t;
}

Vector3f interpolate(const Vector3f& p0, const Vector3f& p1, float t)
{
    return p0 + (p1 - p0) * t;
}

// nPoints: Number of result points to produce, per segment.
vector<Vector2f> evalBSplineCurveCubic(vector<Vector2f> allPoints, int nPoints)
{
    vector<Vector2f> points;

    // sliding window of 4 control points
    int n = allPoints.size() - 3;

    for (int wi = 0; wi < n; wi++)
    {
        // Control points are interpolations on the segments.
        vector<Vector2f> bezierControlPoints;
        Vector2f p1 = interpolate(allPoints[wi + 0], allPoints[wi + 1], 2.0 / 3.0);
        Vector2f p2 = interpolate(allPoints[wi + 1], allPoints[wi + 2], 1.0 / 3.0);
        Vector2f p3 = interpolate(allPoints[wi + 1], allPoints[wi + 2], 2.0 / 3.0);
        Vector2f p4 = interpolate(allPoints[wi + 2], allPoints[wi + 3], 1.0 / 3.0);

        bezierControlPoints.push_back(interpolate(p1, p2, 0.5));
        bezierControlPoints.push_back(p2);
        bezierControlPoints.push_back(p3);
        bezierControlPoints.push_back(interpolate(p3, p4, 0.5));
        evalBezierCurveCubic(bezierControlPoints, nPoints, points);
    }

    return points;
}

void tryBezierCurve()
{
    Vector2f p0(0, 0);
    Vector2f p1(1, 2);
    Vector2f p2(2, 2);
    Vector2f p3(3, 1);
    vector<Vector2f> points = { p0, p1, p2, p3 };

    vector<Vector2f> evalPoints;
    evalBezierCurveCubic(points, 100, evalPoints);
    cv::Mat plotImg = plotPointsAndCurve("Bezier Points", points, evalPoints);
    saveDebugImage(plotImg, "bezier-points");
}

void tryBSplineCurve()
{
    Vector2f p0(0, 0);
    Vector2f p1(1, 2);
    Vector2f p2(2, 2);
    Vector2f p3(3, 1);
    Vector2f p4(4, 1);
    Vector2f p5(5, 2);
    Vector2f p6(6, 0);
    Vector2f p7(7, 0);
    Vector2f p8(8, 0);
    vector<Vector2f> points = { p0, p1, p2, p3, p4, p5, p6, p7, p8 };

    auto evalPoints = evalBSplineCurveCubic(points, 100);
    cv::Mat plotImg = plotPointsAndCurve("B-spline Points", points, evalPoints);
    saveDebugImage(plotImg, "b-spline-points");
}

// Cubic Bezier polynomial
float B(int i, float t) 
{
    float u = 1 - t;
    switch (i) {
    case 0: return u*u*u;
    case 1: return 3*t*u*u;
    case 2: return 3*t*t*u;
    case 3: return t*t*t;
    }
    return 0; // Should not reach here
}

// Eval the bezier surface at a single point.
// controlPoints is uniform 4x4 matrix
// u,v are [0,1]
Vector3f evalBezierSurfaceCubic(const vector<vector<Vector3f>>& controlPoints, float u, float v) 
{
    Vector3f point(0, 0, 0);

    for (int i = 0; i < 4; ++i) 
    {
        for (int j = 0; j < 4; ++j) 
        {
            point += B(i, u) * B(j, v) * controlPoints[i][j];
        }
    }

    return point;
}

Vector3f evalBezierSurfaceCubicUnrolled(const vector<vector<Vector3f>>& controlPoints, float u, float v) 
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
        float a = B(i, u) * v13;
        x += a * controlPoints[i][0].x();
        y += a * controlPoints[i][0].y();
        z += a * controlPoints[i][0].z();

        a = B(i, u) * v12;
        x += a * controlPoints[i][1].x();
        y += a * controlPoints[i][1].y();
        z += a * controlPoints[i][1].z();

        a = B(i, u) * v21;
        x += a * controlPoints[i][2].x();
        y += a * controlPoints[i][2].y();
        z += a * controlPoints[i][2].z();

        a = B(i, u) * v3;
        x += a * controlPoints[i][3].x();
        y += a * controlPoints[i][3].y();
        z += a * controlPoints[i][3].z();
    }

    //// See how it looks to leave out x,y interpolation
    //x = controlPoints[1][1].x() + v;
    //y = controlPoints[1][1].y() + u;

    return Vector3f(x, y, z);
}

void dumpPoints(const char* title, const vector<vector<Vector3f>>& points)
{
    fmt::print("---------------------\n");
    fmt::print("{0}\n", title);

    for (int yi = 0; yi < points.size(); yi++)
    {
        for (int xi = 0; xi < points[yi].size(); xi++)
        {
            fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", points[yi][xi].x(), points[yi][xi].y(), points[yi][xi].z());
        }

        fmt::print("\n");
    }
}

// nPoints: each dimension, so total output points length nPoints^2
// yStart: Index to start putting results into outputPoints. This will enlarge the outer (Y) vector if needed.
// outputPoints: row/y-major matrix
void evalBezierSurfaceCubic(const vector<vector<Vector3f>>& controlPoints, int nPoints, int yStart, vector<vector<Vector3f>>& outputPoints)
{
    for (int yi = 0; yi < nPoints; yi++)
    {
        float v = (float)yi / nPoints;
        if (yStart + yi >= outputPoints.size()) outputPoints.push_back(vector<Vector3f>());

        for (int xi = 0; xi < nPoints; xi++)
        {
            float u = (float)xi / nPoints;
            Vector3f p = evalBezierSurfaceCubicUnrolled(controlPoints, v, u);
            outputPoints[yStart + yi].push_back(p);
        }
    }
}

// One pixel per point.
// This expects equal dim (number of points) in x and y.
cv::Mat plotSurface(const string& chartTitle, const vector<Vector3f>& points, float scale)
{
    float xmin = 0;
    float xmax = 0;
    float ymin = 0;
    float ymax = 0;

    for (const auto& p : points)
    {
        xmin = min(xmin, p.x());
        xmax = max(xmax, p.x());
        ymin = min(ymin, p.y());
        ymax = max(ymax, p.y());
    }

    int dim = lroundf(sqrtf(points.size()));
    int drawHeight = dim;
    int drawWidth = dim;

    float xscale = (drawWidth - 1) / (xmax - xmin);
    float yscale = (drawHeight - 1) / (ymax - ymin);

    cv::Mat img;
    img.create(drawHeight, drawWidth, CV_8UC1);

    for (const auto& p : points)
    {
        int x = (int)lroundf((p.x() - xmin) * xscale);
        int y = (int)lroundf((p.y() - ymin) * yscale);
        img.at<uint8_t>(y, x) = (uint8_t)(p.z() * scale);
    }

    return img;
}

//// Hacked up using Eigen matrices because that's the example I found.
//void gnuPlot3dSurface(const string& chartTitle, const vector<Vector3f>& points)
//{
//    Gnuplot gp;
//    int dim = lroundf(sqrtf(points.size()));
//    //gp << "set zrange [-1:1]\n";
//    gp << "set hidden3d nooffset\n";
//    auto plots = gp.splotGroup();
//
//    //vector<tuple<float, float, float>> tpts;
//    //vector<float> xs, ys, zs;
//    //for (const auto& p : points)
//    //{
//    //    //tpts.push_back(std::make_tuple(p.x(), p.y(), p.z()));
//    //    xs.push_back(p.x());
//    //    ys.push_back(p.y());
//    //    zs.push_back(p.z());
//    //}
//    ////plots.add_plot2d(tpts, "title '" + chartTitle + "'");
//    //plots.add_plot2d(tuple{ xs, ys, zs }, "title '" + chartTitle + "'");
//
//    //Eigen::MatrixXd ptsx(dim, dim);
//    //Eigen::MatrixXd ptsy(dim, dim);
//    //Eigen::MatrixXd ptsz(dim, dim);
//    //int shift = 0;
//
//    //float xmin = FLT_MAX;
//    //float xmax = -FLT_MAX;
//    //float ymin = FLT_MAX;
//    //float ymax = -FLT_MAX;
//
//    //for (const auto& p : points)
//    //{
//    //    xmin = min(xmin, p.x());
//    //    xmax = max(xmax, p.x());
//    //    ymin = min(ymin, p.y());
//    //    ymax = max(ymax, p.y());
//    //}
//
//    //float xscale = (dim - 1) / (xmax - xmin);
//    //float yscale = (dim - 1) / (ymax - ymin);
//
//    //for (const auto& p : points)
//    //{
//    //    int x = (int)lroundf((p.x() - xmin) * xscale);
//    //    int y = (int)lroundf((p.y() - ymin) * yscale);
//    //    ptsx(y, x) = p.x();
//    //    ptsy(y, x) = p.y();
//    //    ptsz(y, x) = p.z();
//    //}
//
//    //for (int yi = 0; yi < dim; yi++)
//    //{
//    //    for (int xi = 0; xi < dim; xi++)
//    //    {
//    //        fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", ptsx(yi, xi), ptsy(yi, xi), ptsz(yi, xi));
//    //    }
//    //    fmt::print("\n");
//    //}
//
//    //// "with lines" looks like a wireframe but requires points in row/col order.
//    //plots.add_plot2d(tuple{ptsx, ptsy, ptsz}, "title '" + chartTitle + "'");
//
//    gp << plots;
//    std::cout << "Press enter to exit." << std::endl;
//    std::cin.get();
//}

void gnuPlot3dSurface(const string& chartTitle, const vector<vector<Vector3f>>& points)
{
    Gnuplot gp;
    gp << "set hidden3d nooffset\n";
    gp << "set terminal wxt size 1400,1000\n";
    auto plots = gp.splotGroup();

    int ydim = (int)points.size();
    int xdim = (int)points[0].size();

    // Hacked up using Eigen matrices because that's the example I found.
    //Eigen::MatrixXd ptsx(ydim, xdim);
    //Eigen::MatrixXd ptsy(ydim, xdim);
    //Eigen::MatrixXd ptsz(ydim, xdim);

    //for (int yi = 0; yi < ydim; yi++)
    //{
    //    for (int xi = 0; xi < xdim; xi++)
    //    {
    //        ptsx(yi, xi) = points[yi][xi].x();
    //        ptsy(yi, xi) = points[yi][xi].y();
    //        ptsz(yi, xi) = points[yi][xi].z();
    //    }
    //}

    //plots.add_plot2d(tuple{ptsx, ptsy, ptsz}, "with lines title '" + chartTitle + "'");

    //for (int yi = 0; yi < ydim; yi++)
    //{
    //    for (int xi = 0; xi < xdim; xi++)
    //    {
    //        fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", ptsx(yi, xi), ptsy(yi, xi), ptsz(yi, xi));
    //    }
    //    fmt::print("\n");
    //}

    std::vector<std::vector<std::tuple<float,float,float>>> pts(ydim);

    for(int yi=0; yi<ydim; yi++) 
    {
        pts[yi].resize(xdim);

        for(int xi=0; xi<xdim; xi++) 
        {
            pts[yi][xi] = std::make_tuple(
                points[yi][xi].x(),
                points[yi][xi].y(),
                points[yi][xi].z()
            );
        }
    }

    plots.add_plot2d(pts, "with lines title '" + chartTitle + "'");

    for (int yi = 0; yi < ydim; yi++)
    {
        for (int xi = 0; xi < xdim; xi++)
        {
            fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", std::get<0>(pts[yi][xi]), std::get<1>(pts[yi][xi]), std::get<2>(pts[yi][xi]));
        }
        fmt::print("\n");
    }

    gp << plots;
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();
}

void tryBezierSurface()
{
    vector<vector<Vector3f>> controlPoints = 
    {
        {Vector3f(0, 0, 0), Vector3f(1, 0, 2), Vector3f(2, 0, 2), Vector3f(3, 0, 1), },
        {Vector3f(0, 1, 1), Vector3f(1, 1, 3), Vector3f(2, 1, 2), Vector3f(3, 1, 3), },
        {Vector3f(0, 2, 1), Vector3f(1, 2, 2), Vector3f(2, 2, 2), Vector3f(3, 2, 2), },
        {Vector3f(0, 3, 0), Vector3f(1, 3, 1), Vector3f(2, 3, 2), Vector3f(3, 3, 1), },
    };

    vector<vector<Vector3f>> evalPoints;
    evalBezierSurfaceCubic(controlPoints, 10, 0, evalPoints);
    //cv::Mat plotImg = plotSurface("Bezier Surface", evalPoints, 10.0f);
    //saveDebugImage(plotImg, "bezier-points");
    gnuPlot3dSurface("Bezier Surface", evalPoints);
}

// controlPoints: have to be in order.
// nPoints: each dimension, so total output points length nPoints^2
void evalBSplineSurfaceCubic(const vector<vector<Vector3f>>& controlPoints, int nPoints, vector<vector<Vector3f>>& outputPoints)
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
            vector<vector<Vector3f>> windowControlPointsX;

            for (int wyi = 0; wyi < 4; wyi++)
            {
                int y = yi + wyi;
                Vector3f p1 = interpolate(controlPoints[y][xi + 0], controlPoints[y][xi + 1], 2.0 / 3.0);
                Vector3f p2 = interpolate(controlPoints[y][xi + 1], controlPoints[y][xi + 2], 1.0 / 3.0);
                Vector3f p3 = interpolate(controlPoints[y][xi + 1], controlPoints[y][xi + 2], 2.0 / 3.0);
                Vector3f p4 = interpolate(controlPoints[y][xi + 2], controlPoints[y][xi + 3], 1.0 / 3.0);

                vector<Vector3f> rowPoints;
                rowPoints.push_back(interpolate(p1, p2, 0.5));
                rowPoints.push_back(p2);
                rowPoints.push_back(p3);
                rowPoints.push_back(interpolate(p3, p4, 0.5));
                windowControlPointsX.push_back(rowPoints);
            }

            dumpPoints("Window Points X", windowControlPointsX);

            // Interpolate in y dir (on the already x-interpolated points, just 4x4 at this point).
            vector<vector<Vector3f>> windowControlPoints;
            for (int wxi = 0; wxi < 4; wxi++) windowControlPoints.push_back(vector<Vector3f>());

            for (int wxi = 0; wxi < 4; wxi++)
            {
                Vector3f p1 = interpolate(windowControlPointsX[0][wxi], windowControlPointsX[1][wxi], 2.0 / 3.0);
                Vector3f p2 = interpolate(windowControlPointsX[1][wxi], windowControlPointsX[2][wxi], 1.0 / 3.0);
                Vector3f p3 = interpolate(windowControlPointsX[1][wxi], windowControlPointsX[2][wxi], 2.0 / 3.0);
                Vector3f p4 = interpolate(windowControlPointsX[2][wxi], windowControlPointsX[3][wxi], 1.0 / 3.0);

                windowControlPoints[0].push_back(interpolate(p1, p2, 0.5));
                windowControlPoints[1].push_back(p2);
                windowControlPoints[2].push_back(p3);
                windowControlPoints[3].push_back(interpolate(p3, p4, 0.5));
            }

            dumpPoints("Window Interpolated Points", windowControlPoints);
            evalBezierSurfaceCubic(windowControlPoints, nPoints, yi * nPoints, outputPoints);
            dumpPoints("Eval Points", outputPoints);
        }
        //return;
    }
}

void tryBSplineSurface()
{
    vector<vector<Vector3f>> controlPoints = 
    {
        {Vector3f(10, 0, 0), Vector3f(11, 0, 2), Vector3f(12, 0, 2), Vector3f(13, 0, 1), Vector3f(14, 0, 2), },
        {Vector3f(10, 1, 1), Vector3f(11, 1, 3), Vector3f(12, 1, 2), Vector3f(13, 1, 3), Vector3f(14, 1, 2), },
        {Vector3f(10, 2, 1), Vector3f(11, 2, 2), Vector3f(12, 2, 2), Vector3f(13, 2, 2), Vector3f(14, 2, 1), },
        {Vector3f(10, 3, 0), Vector3f(11, 3, 1), Vector3f(12, 3, 2), Vector3f(13, 3, 1), Vector3f(14, 3, 2), },
        {Vector3f(10, 4, 2), Vector3f(11, 4, 2), Vector3f(12, 4, 1), Vector3f(13, 4, 2), Vector3f(14, 4, 3), },
    };

    //{Vector3f(0, 0, 0), Vector3f(1, 0, 2), Vector3f(2, 0, 2), Vector3f(3, 0, 1), Vector3f(4, 0, 2), Vector3f(5, 0, 3), },
    //{Vector3f(0, 1, 1), Vector3f(1, 1, 3), Vector3f(2, 1, 2), Vector3f(3, 1, 3), Vector3f(4, 1, 2), Vector3f(5, 1, 1), },
    //{Vector3f(0, 2, 1), Vector3f(1, 2, 2), Vector3f(2, 2, 2), Vector3f(3, 2, 2), Vector3f(4, 2, 1), Vector3f(5, 2, 2), },
    //{Vector3f(0, 3, 0), Vector3f(1, 3, 1), Vector3f(2, 3, 2), Vector3f(3, 3, 1), Vector3f(4, 3, 2), Vector3f(5, 3, 1), },
    //{Vector3f(0, 4, 2), Vector3f(1, 4, 2), Vector3f(2, 4, 1), Vector3f(3, 4, 2), Vector3f(4, 4, 3), Vector3f(5, 4, 1), },
    //{Vector3f(0, 5, 1), Vector3f(1, 5, 1), Vector3f(2, 5, 2), Vector3f(3, 5, 1), Vector3f(4, 5, 1), Vector3f(5, 5, 3), },

    dumpPoints("Control Points", controlPoints);

    vector<vector<Vector3f>> evalPoints;
    evalBSplineSurfaceCubic(controlPoints, 10, evalPoints);
    //cv::Mat plotImg = plotSurface("B-Spline Surface", evalPoints, 10.0f);
    //saveDebugImage(plotImg, "b-spline-points");
    gnuPlot3dSurface("B-Spline Surface", evalPoints);
}

int main()
{
    fmt::print("Starting...\n");
    ImageUtil::init();

    //// load test image
    //string imgPath = "Z:\\TestMedia\\Images\\16u\\000_0_image_2k.png";
    //cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    //saveDebugImage(img, "orig");

    //// create a cvplot
    //vector<float> xs = { 0, 1, 2, 3, 4, 5 };
    //vector<float> ys = { 0, 1, 2, 1, 3, 1 };
    //cv::Mat plotImg = renderPlot("Test Plot", xs, ys);
    //saveDebugImage(plotImg, "plot");

    //tryBezierCurve();
    //tryBSplineCurve();
    //tryGnuPlot();
    //tryBezierSurface();
    tryBSplineSurface();

    fmt::print("Done.\n");
    return 0;
}
