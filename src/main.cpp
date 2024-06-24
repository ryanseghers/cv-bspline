#include <string>
#include <chrono>
#include <ctime>
#include <future>
#include <optional>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "ImageUtil.h"

#include "CvPlotUtil.h"
#include "MiscUtil.h"
#include "BSplineMiscUtil.h"
#include "GnuPlotUtil.h"
#include "BezierVector.h"
#include "BezierMat.h"
#include "BSplineVector.h"
#include "BSplineMat.h"
#include "BSplineGrid.h"
#include "MatWave.h"
#include "ImageTransformBSpline.h"
#include "cudaUtil.h"
#include "cudaBSplineCoeffs.h"
#include "cudaBSplineEval.h"
#include "cudaBSplineTransform.h"

using namespace std;
using namespace std::chrono;
using namespace CppOpenCVUtil;
using namespace CvImageDeform;

const std::string TestImagePath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";

// don't want to depend on windows just for this
const int screenWidth = 2560;
const int screenHeight = 1440;


void tryBezierCurve()
{
    cv::Point2f p0(0, 0);
    cv::Point2f p1(1, 2);
    cv::Point2f p2(2, 2);
    cv::Point2f p3(3, 1);
    vector<cv::Point2f> points = { p0, p1, p2, p3 };

    vector<cv::Point2f> evalPoints;
    evalBezierCurveCubic(points, 100, evalPoints);
    cv::Mat plotImg = plotPointsAndCurve("Bezier Points", points, evalPoints);
    saveDebugImage(plotImg, "bezier-points");
}

void tryBSplineCurve()
{
    cv::Point2f p0(0, 0);
    cv::Point2f p1(1, 2);
    cv::Point2f p2(2, 2);
    cv::Point2f p3(3, 1);
    cv::Point2f p4(4, 1);
    cv::Point2f p5(5, 2);
    cv::Point2f p6(6, 0);
    cv::Point2f p7(7, 0);
    cv::Point2f p8(8, 0);
    vector<cv::Point2f> points = { p0, p1, p2, p3, p4, p5, p6, p7, p8 };

    auto evalPoints = evalBSplineCurveCubic(points, 100);
    cv::Mat plotImg = plotPointsAndCurve("B-spline Points", points, evalPoints);
    saveDebugImage(plotImg, "b-spline-points");
}

/**
 * @brief Eval gaussian with mean 0 and sigma 1.
 */
float gaussianf(float x) 
{
    float sigma = 1.0f; // Standard deviation
    return std::expf(-0.5f * std::powf(x / sigma, 2));
}

/**
* @brief Using the gaussian function in a geometric sense, just for its shape.
* @param halfWidth Half width of peak.
* @param height Max height of peak.
* @param center X center of peak.
* @param x Evaluate the function at this x.
*/
float gaussianScaledf(float halfWidth, float height, float center, float x)
{
    float xOffset = x - center;
    float xScaled = xOffset / (halfWidth / 2);
    float g = gaussianf(xScaled) * height;
    return g;
}

void tryBezierCurveFit()
{
    // Fit curve via three points
    cv::Point2f p0(0, 0);
    cv::Point2f p1(1.5f, 1.6f);
    cv::Point2f p2(3, 1);
    vector<cv::Point2f> points = { p0, p1, p2 };

    vector<cv::Point2f> fittedPoints = fitBezierCurveCubic(points);

    vector<cv::Point2f> evalPoints;
    evalBezierCurveCubic(fittedPoints, 100, evalPoints);
    cv::Mat plotImg = plotPointsAndCurve("Bezier Points", fittedPoints, evalPoints);
    saveDebugImage(plotImg, "bezier-points");
}

// actual fitting is relatively difficult for arbitrary points and can tie the b-spline in knots.
// just using values as b-spline control points under-fits but works quite nicely (at least for uniformly spaced points)
void tryBSplineCurveFit()
{
    vector<cv::Point2f> inputPoints;

    // define input points via gaussian function
    float xSpacing = 1.0f;
    int n = 50;
    float xCenter = ((n - 1) * xSpacing) / 2;

    //for (float x = 0.0f; x < n; x += xSpacing)
    //{
    //    float dx = x - xCenter;
    //    float y = gaussianf(2 * dx / n);
    //    inputPoints.push_back(cv::Point2f(x, y));
    //}

    // define input points via sin function
    //for (float x = 0.0f; x < n; x += xSpacing)
    //{
    //    float y = sinf(x / 20 * 3.14f * 2.0f);
    //    inputPoints.push_back(cv::Point2f(x, y));
    //}

    // some canned points
    float x = 0.0f;
    inputPoints.push_back(cv::Point2f(x++, 1.0f));
    inputPoints.push_back(cv::Point2f(x++, 3.0f));
    inputPoints.push_back(cv::Point2f(x++, 1.0f));
    inputPoints.push_back(cv::Point2f(x++, 1.0f));
    inputPoints.push_back(cv::Point2f(x++, 6.0f));
    inputPoints.push_back(cv::Point2f(x++, -1.0f));
    inputPoints.push_back(cv::Point2f(x++, 1.0f));
    inputPoints.push_back(cv::Point2f(x++, 2.0f));
    inputPoints.push_back(cv::Point2f(x++, 1.0f));

    // eval the inputs directly as b-spline control points
    auto evalPoints = evalBSplineCurveCubic(inputPoints, 100);
    cv::Mat plotImg = plotPointsAndCurve("Input Points as B-spline Control Points", inputPoints, evalPoints);
    saveDebugImage(plotImg, "b-spline-points");

    // fitting
    vector<cv::Point2f> fittedPoints = fitBSplineCurveCubic(inputPoints, 3);

    evalPoints = evalBSplineCurveCubic(fittedPoints, 100);
    plotImg = plotPointsAndCurve("Fitted B-spline Points", inputPoints, evalPoints);
    saveDebugImage(plotImg, "b-spline-points-fitted");
}

void tryBezierSurface()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(0, 0, 0), cv::Point3f(1, 0, 2), cv::Point3f(2, 0, 2), cv::Point3f(3, 0, 1), },
        {cv::Point3f(0, 1, 1), cv::Point3f(1, 1, 3), cv::Point3f(2, 1, 2), cv::Point3f(3, 1, 3), },
        {cv::Point3f(0, 2, 1), cv::Point3f(1, 2, 2), cv::Point3f(2, 2, 2), cv::Point3f(3, 2, 2), },
        {cv::Point3f(0, 3, 0), cv::Point3f(1, 3, 1), cv::Point3f(2, 3, 2), cv::Point3f(3, 3, 1), },
    };

    vector<vector<cv::Point3f>> evalPoints;
    evalBezierSurfaceCubic(controlPoints, 10, 0, evalPoints);
    gnuPlot3dSurface("Bezier Surface", evalPoints);
}

vector<vector<cv::Point3f>> getExampleBSpline5x5ControlPoints()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(10, 0, 0), cv::Point3f(11, 0, 2), cv::Point3f(12, 0, 2), cv::Point3f(13, 0, 1), cv::Point3f(14, 0, 2), },
        {cv::Point3f(10, 1, 1), cv::Point3f(11, 1, 3), cv::Point3f(12, 1, 2), cv::Point3f(13, 1, 3), cv::Point3f(14, 1, 2), },
        {cv::Point3f(10, 2, 1), cv::Point3f(11, 2, 2), cv::Point3f(12, 2, 2), cv::Point3f(13, 2, 2), cv::Point3f(14, 2, 1), },
        {cv::Point3f(10, 3, 0), cv::Point3f(11, 3, 1), cv::Point3f(12, 3, 2), cv::Point3f(13, 3, 1), cv::Point3f(14, 3, 2), },
        {cv::Point3f(10, 4, 2), cv::Point3f(11, 4, 2), cv::Point3f(12, 4, 1), cv::Point3f(13, 4, 2), cv::Point3f(14, 4, 3), },
    };

    return controlPoints;
}

vector<vector<cv::Point3f>> getExampleBSpline4x4ControlPoints()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(10, 0, 0), cv::Point3f(11, 0, 2), cv::Point3f(12, 0, 2), cv::Point3f(13, 0, 1), },
        {cv::Point3f(10, 1, 1), cv::Point3f(11, 1, 3), cv::Point3f(12, 1, 2), cv::Point3f(13, 1, 3), },
        {cv::Point3f(10, 2, 1), cv::Point3f(11, 2, 2), cv::Point3f(12, 2, 2), cv::Point3f(13, 2, 2), },
        {cv::Point3f(10, 3, 0), cv::Point3f(11, 3, 1), cv::Point3f(12, 3, 2), cv::Point3f(13, 3, 1), },
    };

    return controlPoints;
}

void tryBSplineSurface()
{
    vector<vector<cv::Point3f>> controlPoints = getExampleBSpline5x5ControlPoints();

    //{cv::Point3f(0, 0, 0), cv::Point3f(1, 0, 2), cv::Point3f(2, 0, 2), cv::Point3f(3, 0, 1), cv::Point3f(4, 0, 2), cv::Point3f(5, 0, 3), },
    //{cv::Point3f(0, 1, 1), cv::Point3f(1, 1, 3), cv::Point3f(2, 1, 2), cv::Point3f(3, 1, 3), cv::Point3f(4, 1, 2), cv::Point3f(5, 1, 1), },
    //{cv::Point3f(0, 2, 1), cv::Point3f(1, 2, 2), cv::Point3f(2, 2, 2), cv::Point3f(3, 2, 2), cv::Point3f(4, 2, 1), cv::Point3f(5, 2, 2), },
    //{cv::Point3f(0, 3, 0), cv::Point3f(1, 3, 1), cv::Point3f(2, 3, 2), cv::Point3f(3, 3, 1), cv::Point3f(4, 3, 2), cv::Point3f(5, 3, 1), },
    //{cv::Point3f(0, 4, 2), cv::Point3f(1, 4, 2), cv::Point3f(2, 4, 1), cv::Point3f(3, 4, 2), cv::Point3f(4, 4, 3), cv::Point3f(5, 4, 1), },
    //{cv::Point3f(0, 5, 1), cv::Point3f(1, 5, 1), cv::Point3f(2, 5, 2), cv::Point3f(3, 5, 1), cv::Point3f(4, 5, 1), cv::Point3f(5, 5, 3), },

    dumpPoints("Control Points", controlPoints);

    vector<vector<cv::Point3f>> evalPoints;
    evalBSplineSurfaceCubic(controlPoints, 10, evalPoints);
    //cv::Mat plotImg = plotSurface("B-Spline Surface", evalPoints, 10.0f);
    //saveDebugImage(plotImg, "b-spline-points");
    gnuPlot3dSurface("B-Spline Surface", evalPoints);
}

void tryComputeBezierControlPoints()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(10, 0, 0), cv::Point3f(11, 0, 2), cv::Point3f(12, 0, 1) },
        {cv::Point3f(10, 1, 1), cv::Point3f(11, 1, 3), cv::Point3f(12, 1, 2) },
        {cv::Point3f(10, 2, 1), cv::Point3f(11, 2, 2), cv::Point3f(12, 2, 1) },
    };

    cv::Mat zs = pointsToMatZs(controlPoints);
    dumpMat("B-Spline Control Points", zs);

    cv::Mat bezierControlMatZs;
    computeBezierControlPoints(zs, bezierControlMatZs, false);

    auto bezierControlPoints = matToPoints(bezierControlMatZs);
    dumpPoints("Bezier Control Points", bezierControlPoints);
    gnuPlot3dSurfaceZs("Bezier Control Points", bezierControlMatZs);
}

void tryBezierSurfacePoint()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(0, 0, 0), cv::Point3f(1, 0, 2), cv::Point3f(2, 0, 2), cv::Point3f(3, 0, 1), },
        {cv::Point3f(0, 1, 1), cv::Point3f(1, 1, 3), cv::Point3f(2, 1, 2), cv::Point3f(3, 1, 3), },
        {cv::Point3f(0, 2, 1), cv::Point3f(1, 2, 2), cv::Point3f(2, 2, 2), cv::Point3f(3, 2, 2), },
        {cv::Point3f(0, 3, 0), cv::Point3f(1, 3, 1), cv::Point3f(2, 3, 2), cv::Point3f(3, 3, 1), },
    };

    cv::Mat controlMatZs = pointsToMatZs(controlPoints);

    // Single point
    cv::Point3f pt = evalBezierSurfaceCubicPointMat(controlMatZs, 0.1f, 0.2f);
    fmt::print("Point: {0}, {1}, {2}\n", pt.x, pt.y, pt.z);

    // simd
    pt = evalBezierSurfaceCubicPointMatSimd(controlMatZs, 0.1f, 0.2f);
    fmt::print("Point: {0}, {1}, {2}\n", pt.x, pt.y, pt.z);

    // simd
    pt = evalBezierSurfaceCubicPointMatSimd2(controlMatZs, 0.1f, 0.2f);
    fmt::print("Point: {0}, {1}, {2}\n", pt.x, pt.y, pt.z);
}

void tryBezierSurfaceMat()
{
    vector<vector<cv::Point3f>> controlPoints = 
    {
        {cv::Point3f(0, 0, 0), cv::Point3f(1, 0, 2), cv::Point3f(2, 0, 2), cv::Point3f(3, 0, 1), },
        {cv::Point3f(0, 1, 1), cv::Point3f(1, 1, 3), cv::Point3f(2, 1, 2), cv::Point3f(3, 1, 3), },
        {cv::Point3f(0, 2, 1), cv::Point3f(1, 2, 2), cv::Point3f(2, 2, 2), cv::Point3f(3, 2, 2), },
        {cv::Point3f(0, 3, 0), cv::Point3f(1, 3, 1), cv::Point3f(2, 3, 2), cv::Point3f(3, 3, 1), },
    };

    cv::Mat controlMatZs = pointsToMatZs(controlPoints);

    // Surface
    cv::Mat evalMat;
    int pointsDim = 10;
    evalMat.create(pointsDim, pointsDim, CV_32FC3);
    evalBezierSurfaceCubicMatAvx(controlMatZs, 10, 0.0f, 0.0f, evalMat);
    dumpMat("Bezier Surface", evalMat);
    gnuPlot3dSurfaceMat("Bezier Surface", evalMat);
}

void tryBSplineSurfaceMat()
{
    vector<vector<cv::Point3f>> controlPoints = getExampleBSpline5x5ControlPoints();
    dumpPoints("Vector B-Spline Control Points", controlPoints);
    int nPointsDim = 10;

    //// Vector for comparison
    //vector<vector<cv::Point3f>> evalPoints;
    //evalBSplineSurfaceCubic(controlPoints, nPointsDim, evalPoints, true);
    //gnuPlot3dSurface("B-Spline Surface", evalPoints);
    //dumpPoints("Vector B-Spline Surface", evalPoints);

    // Mat
    cv::Mat zs = pointsToMatZs(controlPoints);
    dumpMat("Mat B-Spline Control Points", zs);

    cv::Mat bezierControlMatZs;
    computeBezierControlPoints(zs, bezierControlMatZs, true);
    dumpMat("Mat Bezier Control Points", bezierControlMatZs);

    //auto bezierControlPoints = matToPoints(bezierControlMat);
    //dumpPoints("Bezier Control Points", bezierControlPoints);

    //gnuPlot3dSurfaceZs("Bezier Control Points", bezierControlMatZs);

    // eval and plot
    //vector<vector<cv::Point3f>> evalPoints;
    //evalBSplineSurfaceCubic2(bezierControlMat, 10, evalPoints);
    //dumpPoints("B-Spline Surface", evalPoints);
    //gnuPlot3dSurface("B-Spline Surface", evalPoints);

    // eval and plot Mat
    cv::Mat evalMat;
    evalBSplineSurfaceCubicPrecomputedMat(bezierControlMatZs, nPointsDim, evalMat);

    // The computable area is smaller than the control points area.
    cv::Rect roi(1 * nPointsDim, 1 * nPointsDim, (zs.cols - 3) * nPointsDim, (zs.rows - 3) * nPointsDim);
    cv::Mat evalComputableArea = evalMat(roi);
    dumpMat("Mat B-Spline Surface", evalComputableArea);
    gnuPlot3dSurfaceMat("B-Spline Surface", evalComputableArea);
}

void tryBSplineSurfaceMat2()
{
    vector<vector<cv::Point3f>> controlPoints = getExampleBSpline5x5ControlPoints();
    dumpPoints("Vector B-Spline Control Points", controlPoints);
    int nPointsDim = 10;

    // Mat
    cv::Mat zs = pointsToMatZs(controlPoints);
    dumpMat("Mat B-Spline Control Points", zs);

    // eval and plot Mat
    cv::Mat evalMat;
    evalBSplineSurfaceCubicMat(zs, nPointsDim, evalMat);

    // The computable area is smaller than the control points area.
    cv::Rect roi(1 * nPointsDim, 1 * nPointsDim, (zs.cols - 3) * nPointsDim, (zs.rows - 3) * nPointsDim);
    cv::Mat evalComputableArea = evalMat(roi);
    dumpMat("Mat B-Spline Surface", evalComputableArea);
    gnuPlot3dSurfaceMat("B-Spline Surface", evalComputableArea);
}

void tryBSplineGrid()
{
    BSplineGrid grid(4, 5);
    grid.fillRandomControlPoints(0, 4);
    cv::Mat evalMat = grid.evalSurface(5);
    gnuPlot3dSurfaceMat("B-Spline Surface", evalMat);
}

void try3fMat()
{
    int rows = 4;
    int cols = 3;

    cv::Mat m1 = cv::Mat::ones(rows, cols, CV_32FC3);
    setDummyValues(m1);
    dumpMat("m1", m1);

    cv::Rect roi(1, 1, 1, 2);
    cv::Mat roiMat = m1(roi);
    setDummyValues(roiMat);

    dumpMat("roiMat", roiMat);
    dumpMat("m1", m1);
}

cv::Mat createMatWithColumnIndices(int rows, int cols) 
{
    cv::Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < mat.cols; ++i) 
    {
        mat.col(i) = i;
    }
    return mat;
}

cv::Mat createMatWithRowIndices(int rows, int cols) 
{
    cv::Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < mat.rows; ++i) 
    {
        mat.row(i) = i;
    }
    return mat;
}

cv::Mat ensureImageDims(cv::Mat img, int sizeMult)
{
    if ((img.rows % sizeMult) || (img.cols % sizeMult))
    {
        cv::Mat tmp;
        int newRows = (img.rows / sizeMult) * sizeMult;
        int newCols = (img.cols / sizeMult) * sizeMult;
        cv::resize(img, tmp, cv::Size(newCols, newRows));
        img = tmp;
    }

    return img;
}

cv::Mat buildTestGridImage(int sizeMult)
{
    cv::Mat img = cv::Mat::zeros(screenHeight, screenWidth, CV_8UC4);
    img = ensureImageDims(img, sizeMult);

    int gridSpacing = sizeMult;

    cv::Scalar white(255, 255, 255);
    cv::Scalar blue(255, 0, 0);
    cv::Scalar black(0, 0, 0);

    img = white;
    cv::Scalar gridColor = black;

    // vert lines
    for (int x = 0; x < img.cols; x += gridSpacing)
    {
        cv::Point2i pt1(x, 0);
        cv::Point2i pt2(x, img.rows);

        cv::line(img, pt1, pt2, gridColor);
    }

    // horz lines
    for (int y = 0; y < img.rows; y += gridSpacing)
    {
        cv::Point2i pt1(0, y);
        cv::Point2i pt2(img.cols, y);

        cv::line(img, pt1, pt2, gridColor);
    }

    return img;
}

cv::Mat loadAndConvertTestImage(const string& imgPath, bool doConvertTo8u = true, int sizeMult = 64)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    cv::Mat tmp;
    //saveDebugImage(img, "orig");

    if (doConvertTo8u && (img.type() != CV_8U))
    {
        cv::cvtColor(img, tmp, cv::COLOR_RGB2GRAY);
        img = tmp;
        //saveDebugImage(img, "gray");
    }

    // Downsize for perf
    if (img.rows > screenHeight)
    {
        int newRows = screenHeight;
        int newCols = screenWidth * img.cols / img.rows;
        cv::resize(img, img, cv::Size(newCols, newRows));
        //saveDebugImage(img, "scaled");
    }

    // Make image size a multiple of sizeMult for simplicity
    img = ensureImageDims(img, sizeMult);

    return img;
}

void renderDistortedImage(cv::Mat& img, const cv::Mat& xcoords, const cv::Mat& ycoords, BSplineGrid& grid)
{
    // eval for every pixel in image
    int res = img.rows / grid.rows();
    cv::Mat evalMat = grid.evalSurface(res);
    std::vector<cv::Mat> channels(3);
    cv::split(evalMat, channels);
    cv::Mat evalMatZs = channels[2];
     
    // remap takes x coordinate to sample
    cv::Mat xmap = xcoords + evalMatZs;
    cv::Mat dest;
    //cv::remap(img, dest, xmap, ycoords, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(img, dest, xmap, ycoords, cv::INTER_CUBIC, cv::BORDER_CONSTANT);
    saveDebugImage(dest, "remap");
}

void oscillateControlPoints(cv::Mat& src, int step, cv::Mat& dst)
{
    dst.resize(src.rows, src.cols);

    // done one full cycle in 20 steps
    float s = sin(step * 1 / 20.0f * 6.28f);

    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            //float z = src.at<float>(i, j);
            //z = 10.0f * sin(i * 0.33333f);
            dst.at<float>(i, j) = src.at<float>(i, j) * s;
        }
    }
}

void tryBSplineGridDeformImage()
{
    // load test image
    //string imgPath = "Z:\\TestMedia\\Images\\16u\\000_0_image_2k.png";
    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath);

    // grid
    int pxPerCell = 64;
    int gridRows = img.rows / pxPerCell;
    int gridCols = img.cols / pxPerCell;
    BSplineGrid grid(gridRows, gridCols);
    //grid.fillRandomControlPoints(-10, 10);

    // remap takes x coordinate to sample
    cv::Mat xcoords = createMatWithColumnIndices(img.rows, img.cols);
    cv::Mat ycoords = createMatWithRowIndices(img.rows, img.cols);
    renderDistortedImage(img, xcoords, ycoords, grid);

    // modify the control points to make the distortion move around in the image in waves
    cv::Mat controlMatZs = grid.getControlPointZs();
    controlMatZs.at<float>(controlMatZs.rows / 2, controlMatZs.cols / 2) = 50.0f;
    //saveDebugImage(controlMatZs, "control-points");
    //cv::Mat origZs = controlMatZs.clone();

    //for (int i = 0; i < 40; ++i)
    //{
    //    fmt::println("Step {}", i);
    //    oscillateControlPoints(origZs, i, controlMatZs);
    //    renderDistortedImage(img, xcoords, ycoords, grid);
    //}

    // wave
    float k = 0.1f; // spring constant
    float m = 1.0f; // mass
    float friction = 1.0f - 0.05f;
    MatWave wave(k, m, friction, controlMatZs.rows, controlMatZs.cols);
    int nCycles = 100;

    for (int i = 0; i < nCycles; ++i)
    {
        fmt::println("Step {}", i);
        wave.apply(controlMatZs);
        //saveDebugImage(controlMatZs, "control-points");
        renderDistortedImage(img, xcoords, ycoords, grid);
    }
}

void benchmarkBSplineSurface()
{
    int nCycles;
    int nPointsDim = 128;
    int gridRows = 16;
    int gridCols = 16;
    BSplineGrid grid(gridRows, gridCols);
    grid.fillRandomControlPoints(-10, 10);
    cv::Mat zs = grid.getControlPointZs();
    cv::Mat evalMat, bezierControlPointsZs;

    // computeBezierControlPoints
    nCycles = 100;
    auto start = CppBaseUtil::getTimeNow();

    for (int i = 0; i < nCycles; ++i)
    {
        computeBezierControlPoints(zs, bezierControlPointsZs, false);
    }

    //evalBSplineSurfaceCubicPrecomputedMat(bezierControlPointsZs, nPointsDim, evalMat);

    auto durationSeconds = CppBaseUtil::getDurationSeconds(start);
    fmt::print("computeBezierControlPoints: {0:.1f} ms\n", durationSeconds * 1000.0f);

    // evalBSplineSurfaceCubicPrecomputedMat
    nCycles = 20;
    start = CppBaseUtil::getTimeNow();

    for (int i = 0; i < nCycles; ++i)
    {
        evalBSplineSurfaceCubicPrecomputedMat(bezierControlPointsZs, nPointsDim, evalMat);
    }

    durationSeconds = CppBaseUtil::getDurationSeconds(start);
    fmt::print("evalBSplineSurfaceCubicPrecomputedMat: {0:.1f} ms\n", durationSeconds * 1000.0f / nCycles);
}

void tryMatWave()
{
    cv::Mat mat = cv::Mat::zeros(4, 4, CV_32F);
    MatWave wave(0.1f, 2.0f, 0.9f, mat.rows, mat.cols);
    mat.at<float>(1, 1) = 1.0f;
    dumpMat("Wave", mat);

    for (int i = 0; i < 20; i++)
    {
        wave.apply(mat);
        dumpMat("Wave", mat);
    }
}

void applyGaussianDomeDeformation(ImageTransformBSpline& imageTransform, float halfWidth, cv::Point2f deformCenter,
    cv::Point2f offsetVector, bool doDebug)
{
    float pxPerCell = imageTransform.getPxPerCell();
    cv::Mat dxControlMatZs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dyControlMatZs = imageTransform.getDyGrid().getControlPointZs();

    // Limit the height (magnitude of offset) vs half-width by widening
    float offsetMag = sqrtf(offsetVector.x * offsetVector.x + offsetVector.y * offsetVector.y);
    halfWidth = max(offsetMag, halfWidth);

    cv::Mat xtmp(dxControlMatZs.rows - 2, dxControlMatZs.cols - 2, CV_32F);
    cv::Mat ytmp(dxControlMatZs.rows - 2, dxControlMatZs.cols - 2, CV_32F);

    for (int r = 1; r < dxControlMatZs.rows - 1; r++)
    {
        for (int c = 1; c < dxControlMatZs.cols - 1; c++)
        {
            float cellx = (c - 1) * pxPerCell;
            float celly = (r - 1) * pxPerCell;

            float cellDx = cellx - deformCenter.x;
            float cellDy = celly - deformCenter.y;
            float cellDist = sqrtf(cellDx * cellDx + cellDy * cellDy);

            // use gaussians to decide how to stretch, per axis
            float x1 = gaussianScaledf(halfWidth, offsetVector.x, deformCenter.x, cellx);
            float y1 = gaussianScaledf(halfWidth, offsetVector.y, deformCenter.y, celly);

            // Less offset based on dist from cell to mouse start point
            float distDamper = gaussianScaledf(halfWidth, 1.0f, 0.0f, cellDist);

            float xo = x1 * distDamper;
            float yo = y1 * distDamper;

            dxControlMatZs.at<float>(r, c) = -xo;
            dyControlMatZs.at<float>(r, c) = -yo;

            xtmp.at<float>(r - 1, c - 1) = xo;
            ytmp.at<float>(r - 1, c - 1) = yo;
        }
    }

    if (doDebug)
    {
        saveDebugImage(xtmp, "x-offset-values");
        saveDebugImage(ytmp, "y-offset-values");
    }
}

const int waveBorderWidth = 2; // in cells
int pxPerCell = 64;
ImageTransformBSpline* pImageTransform = nullptr;
MatWave* dxWave = nullptr;
MatWave* dyWave = nullptr;

cv::Mat dxControlMatZsBase;
cv::Mat dyControlMatZsBase;
// As the mouse drags these are modified, then applied to the originals and zero'd when dragging is done
cv::Mat dxControlMatZsOffsets;
cv::Mat dyControlMatZsOffsets;



//cv::Mat dxControlMatZs;
//cv::Mat dyControlMatZs;
bool isMouseDown = false;
cv::Point2f mouseStartPoint, mouseEndPoint;

void mouseCallbackWave(int event, int x, int y, int flags, void* userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        std::cout << "Left button clicked at (" << x << ", " << y << ")" << std::endl;

        cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
        cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();

        // control points are outside of the surface
        int xi = x / pxPerCell;
        int yi = y / pxPerCell;

        if ((xi >= waveBorderWidth) && (xi < dxControlMatZs.cols - 1 - waveBorderWidth * 2)
            && (yi >= waveBorderWidth) && (yi < dyControlMatZs.rows - 1 - waveBorderWidth * 2))
        {
            dxControlMatZs.at<float>(yi + 1, xi + 1) = 50.0f;
            dyControlMatZs.at<float>(yi + 1, xi + 1) = 50.0f;
        }
    }
}

/**
 * @brief Adjust control points based on mouse drag.
 * This is in the sense of a forward/push image remap, from mouse start to end.
 */
void fitMouseDragForward()
{
    // +1 because control points go outside of surface
    int xi = lroundf(mouseStartPoint.x / pxPerCell) + 1;
    int yi = lroundf(mouseStartPoint.y / pxPerCell) + 1;

    cv::Point2f offset = mouseEndPoint - mouseStartPoint;

    // Lock the 4 mouse start points and apply offset
    vector<cv::Point2i> lockedPoints;
    lockedPoints.push_back(cv::Point2i(xi, yi));
    lockedPoints.push_back(cv::Point2i(xi + 1, yi));
    lockedPoints.push_back(cv::Point2i(xi, yi + 1));
    lockedPoints.push_back(cv::Point2i(xi + 1, yi + 1));
    dxWave->setLockedPoints(lockedPoints);
    dyWave->setLockedPoints(lockedPoints);

    cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
    cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();

    for (int i = 0; i < lockedPoints.size(); i++)
    {
        dxControlMatZs.at<float>(lockedPoints[i].y, lockedPoints[i].x) = -offset.x;
        dyControlMatZs.at<float>(lockedPoints[i].y, lockedPoints[i].x) = -offset.y;
    }
}

/**
 * @brief Adjust control points based on mouse drag.
 * This is in the sense of a pull image remap, from mouse end to start.
 */
void fitMouseDrag()
{
    // +1 because control points go outside of surface
    int xsi = int(mouseStartPoint.x / pxPerCell) + 1;
    int ysi = int(mouseStartPoint.y / pxPerCell) + 1;
    int xi = int(mouseEndPoint.x / pxPerCell) + 1;
    int yi = int(mouseEndPoint.y / pxPerCell) + 1;

    cv::Point2f offset = mouseEndPoint - mouseStartPoint;

    // Update the locked points in the wave/spring grid sim
    dxWave->clearLockedPoints();
    dyWave->clearLockedPoints();
    vector<cv::Point2i> lockedPoints;
    lockedPoints.push_back(cv::Point2i(xi, yi));
    lockedPoints.push_back(cv::Point2i(xi + 1, yi));
    lockedPoints.push_back(cv::Point2i(xi, yi + 1));
    lockedPoints.push_back(cv::Point2i(xi + 1, yi + 1));
    dxWave->setLockedPoints(lockedPoints);
    dyWave->setLockedPoints(lockedPoints);

    // Apply this offset
    cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
    cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();

    for (int i = 0; i < lockedPoints.size(); i++)
    {
        dxControlMatZs.at<float>(lockedPoints[i].y, lockedPoints[i].x) = offset.x;
        dyControlMatZs.at<float>(lockedPoints[i].y, lockedPoints[i].x) = offset.y;
    }

    //// represent the offset
    //dxControlMatZsOffsets = 0.0f;
    //dyControlMatZsOffsets = 0.0f;

    //for (int i = 0; i < lockedPoints.size(); i++)
    //{
    //    dxControlMatZsOffsets.at<float>(lockedPoints[i].y, lockedPoints[i].x) = offset.x;
    //    dyControlMatZsOffsets.at<float>(lockedPoints[i].y, lockedPoints[i].x) = offset.y;
    //}

    //// apply this offset to transform
    //cv::Mat newDx = dxControlMatZsBase + dxControlMatZsOffsets;
    //cv::Mat newDy = dyControlMatZsBase + dyControlMatZsOffsets;
    //newDx.copyTo(dxControlMatZs);
    //newDy.copyTo(dyControlMatZs);

    //dxControlMatZs = dxControlMatZsBase + dxControlMatZsOffsets;
    //dyControlMatZs = dyControlMatZsBase + dyControlMatZsOffsets;
}

void mouseCallbackFitting(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        fmt::println("Left button down at: {0}, {1}", x, y);
        mouseStartPoint.x = x;
        mouseStartPoint.y = y;
        isMouseDown = true;

        dxWave->storeLockedPoints();
        dyWave->storeLockedPoints();

        //// Take a clone of the current transform as our base for this drag.
        //cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
        //cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();
        //dxControlMatZsBase = dxControlMatZs.clone();
        //dyControlMatZsBase = dyControlMatZs.clone();

        //// Setup and clear the offsets mats
        //dxControlMatZsOffsets = cv::Mat::zeros(dxControlMatZs.rows, dxControlMatZs.cols, dxControlMatZs.type());
        //dyControlMatZsOffsets = cv::Mat::zeros(dyControlMatZs.rows, dyControlMatZs.cols, dyControlMatZs.type());
    }
    else if (isMouseDown && (event == cv::EVENT_MOUSEMOVE))
    {
        mouseEndPoint.x = x;
        mouseEndPoint.y = y;
        //fmt::println("Drag from: ({0:.1f}, {1:.1f}) to ({2:.1f}, {3:.1f})", mouseStartPoint.x, mouseStartPoint.y, mouseEndPoint.x, mouseEndPoint.y);
        fitMouseDrag();
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        mouseEndPoint.x = x;
        mouseEndPoint.y = y;
        fmt::println("Left button up at: {0}, {1}", x, y);
        isMouseDown = false;
        //fitMouseDrag();

        //cv::Point2f offset = mouseEndPoint - mouseStartPoint;
        //fmt::println("Drag from: ({0:.1f}, {1:.1f}) to ({2:.1f}, {3:.1f})", mouseStartPoint.x, mouseStartPoint.y, mouseEndPoint.x, mouseEndPoint.y);
        //fmt::println("Drag distance: {0:.1f}", sqrtf(offset.x * offset.x + offset.y * offset.y));

        //// Update the main transform matrices
        //cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
        //cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();
        ////dxControlMatZs = dxControlMatZsBase;
        ////dyControlMatZs = dyControlMatZsBase;
        ////dxControlMatZsBase.copyTo(dxControlMatZs);
        //dxControlMatZs.copyTo(dxControlMatZsBase);
        //dyControlMatZs.copyTo(dyControlMatZsBase);
    }
}

// CudaMat is just a wrapper, no memory ownership.
template <typename T>
CudaMat2<T> CreateCudaMat(cv::Mat mat)
{
    CudaMat2<T> cudaMat;
    cudaMat.dataHost = (T*)mat.ptr();
    cudaMat.cols = mat.cols;
    cudaMat.rows = mat.rows;

    int step = mat.step;
    int step1 = mat.step1();

    // for cv::Mat, step[0] is the stride, in bytes
    cudaMat.stride = mat.step[0] / sizeof(T);

    return cudaMat;
}

void showImageTransformBSpline()
{
    bool doCpu = false;
    bool doMouseClickWaves = false;
    bool doWaves = true;
    bool doShowDebugImage = false;

    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC

    cv::Mat img = loadAndConvertTestImage(TestImagePath, false, pxPerCell);
    //cv::Mat img = buildTestGridImage(pxPerCell);
    //saveDebugImage(img, "orig");
    //return;

    pImageTransform = new ImageTransformBSpline(img, pxPerCell);
    cv::Mat dst;

    // wave
    cv::Mat dxControlMatZs = pImageTransform->getDxGrid().getControlPointZs();
    cv::Mat dyControlMatZs = pImageTransform->getDyGrid().getControlPointZs();

    float k = 0.1f; // spring constant
    float m = 1.0f; // mass
    float friction = 1.0f - 0.1f;

    dxWave = new MatWave(k, m, friction, dxControlMatZs.rows, dxControlMatZs.cols);
    dyWave = new MatWave(k, m, friction, dyControlMatZs.rows, dyControlMatZs.cols);

    dxWave->setLockedBorder(waveBorderWidth);
    dyWave->setLockedBorder(waveBorderWidth);

    // initial perturbation
    int xCenter = dxControlMatZs.cols / 2;
    int yCenter = dxControlMatZs.rows / 2;

    // Setup for CUDA
    int deviceId = 0;
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic

    CudaMat2<float> cudaDxs = CreateCudaMat<float>(dxControlMatZs);
    CudaMat2<float> cudaDys = CreateCudaMat<float>(dyControlMatZs);

    cv::Mat imgBgra;
    cv::cvtColor(img, imgBgra, cv::COLOR_BGR2BGRA);

    int bSplineGridRows = pImageTransform->getDxGrid().rows();
    int bSplineGridCols = pImageTransform->getDxGrid().cols();
    float dxScale = (float)imgBgra.cols / bSplineGridCols; // pixels per cell
    float dyScale = (float)imgBgra.rows / bSplineGridRows; // pixels per cell

    // bgra
    CudaMat2<BgraQuad> srcCudaBgra = CreateCudaMat<BgraQuad>(imgBgra);
    cv::Mat dstCudaBgra = cv::Mat::zeros(imgBgra.rows, imgBgra.cols, CV_8UC4);
    CudaMat2<BgraQuad> dstCudaBgra2 = CreateCudaMat<BgraQuad>(dstCudaBgra);

    // interactive window
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Image", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    if (doMouseClickWaves)
    {
        cv::setMouseCallback("Image", mouseCallbackWave);
    }
    else
    {
        cv::setMouseCallback("Image", mouseCallbackFitting);
    }

    while(true)
    {
        if (doWaves)
        {
            dxWave->apply(dxControlMatZs);
            dyWave->apply(dyControlMatZs);
        }

        if (doShowDebugImage)
        {
            //cv::Mat evalImg = pImageTransform->getDxGrid().evalSurface(pxPerCell);
            cv::Mat fieldImg = pImageTransform->getDxGrid().renderField(pxPerCell);
            //cv::cvtColor(dstCudaBgra, dst, cv::COLOR_BGRA2BGR);
            dst = fieldImg;
        }
        else if (doCpu)
        {
            pImageTransform->transformImage(img, interp, dst, false);
        }
        else
        {
            cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);
            cv::cvtColor(dstCudaBgra, dst, cv::COLOR_BGRA2BGR);
        }

        cv::imshow("Image", dst);

        int key = cv::waitKey(1);

        if (key >= 0)
        {
            if (key == 27) // esc
            {
                break;
            }
            fmt::print("Key: {0}", key);

            if (key == 'd')
            {
                // use the cpu transform to save debug images
                pImageTransform->transformImage(img, interp, dst, true);
                cv::Mat evalImg = pImageTransform->getDxGrid().evalSurface(pxPerCell);
                saveDebugImage(evalImg, "eval");
                saveDebugImage(dxControlMatZs, "dxControlMatZs");

                dxWave->saveDebugImages("dxWave");

                //cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);
                //cv::cvtColor(dstCudaBgra, dst, cv::COLOR_BGRA2BGR);
                //saveDebugImage(dst, "deformed");

                //cv::Mat dxSurface = pImageTransform->getDxGrid().evalSurface(pxPerCell);
                //std::vector<cv::Mat> channels;
                //cv::split(dxSurface, channels);
                //cv::Mat dxSurfaceZs = channels[2];

                //fmt::println("dxSurfaceZs: {0}", ImageUtil::getImageDescString(dxSurfaceZs));
                //saveDebugImage(dxSurfaceZs, "dxSurfaceZs");

                //fmt::println("dxControlMatZs: {0}", ImageUtil::getImageDescString(dxControlMatZs));
                //saveDebugImage(dxControlMatZs, "dxControlMatZs");
            }
            else if (key == 'c')
            {
                // clear distortions
                dxWave->clearLockedPoints();
                dxWave->clearStoredLockedPoints();

                dyWave->clearLockedPoints();
                dyWave->clearStoredLockedPoints();
            }
        }
    }

    cv::destroyAllWindows();
}

// Run and compare all the implementations
void tryAllComputeBezierControlPoints()
{
    bool doFull = true;

    BSplineGrid grid(2, 3);
    grid.fillRandomControlPoints(0, 4);
    dumpMat("B-Spline Control Points", grid.getControlPointZs());

    // Simple transpose method
    cv::Mat bezierControlPointsZs;
    computeBezierControlPointsSimple(grid.getControlPointZs(), bezierControlPointsZs);
    dumpMat("CPU Bezier Control Points", bezierControlPointsZs, 3, 3);
    //gnuPlot3dSurfaceMat("Orig CPU B-Spline Surface", bezierControlPointsZs);

    //// Canonical impl
    //cv::Mat bezierControlPointsZsCanonical;
    //computeBezierControlPoints(grid.getControlPointZs(), bezierControlPointsZsCanonical, doFull);
    //dumpMat("CPU Bezier Control Points", bezierControlPointsZsCanonical, 3, 3);
    //printDiffMat(bezierControlPointsZs, bezierControlPointsZsCanonical, 3, 3);

    //// Plain
    //cv::Mat bezierControlPointsZsPlain;
    //computeBezierControlPointsPlain(grid.getControlPointZs(), bezierControlPointsZsPlain);
    //dumpMat("CPU Bezier Control Points (Plain)", bezierControlPointsZsPlain, 3, 3);
    //printDiffMat(bezierControlPointsZs, bezierControlPointsZsPlain, 3, 3);

    // Single cell isolated
    cv::Mat bezierControlPointsZsIsolated;
    computeBezierControlPointsIsolated(grid.getControlPointZs(), bezierControlPointsZsIsolated, doFull);
    dumpMat("CPU Bezier Control Points (Isolated)", bezierControlPointsZsIsolated, 3, 3);
    printDiffMat(bezierControlPointsZs, bezierControlPointsZsIsolated, 3, 3);

    // CUDA
    int deviceId = 0;
    CudaMat2<float> cudaControlPointZs = CreateCudaMat<float>(grid.getControlPointZs());
    cv::Mat cudaResultBezierControlPointsZs = cv::Mat::zeros(bezierControlPointsZs.rows, bezierControlPointsZs.cols, CV_32F);
    CudaMat2<float> cudaBezierControlPointZs = CreateCudaMat<float>(cudaResultBezierControlPointsZs);
    cudaComputeBezierControlPoints(deviceId, cudaControlPointZs, cudaBezierControlPointZs);
    dumpMat("CUDA Bezier Control Points", cudaResultBezierControlPointsZs, 3, 3);
    printDiffMat(bezierControlPointsZs, cudaResultBezierControlPointsZs, 3, 3);
}

// Eval with pre-computed bezier coeffs
void tryCudaEvalBSpline()
{
    BSplineGrid grid(2, 3);
    grid.fillRandomControlPoints(0, 4);
    dumpMat("B-Spline Control Points", grid.getControlPointZs());

    // Compute bezier coeffs
    cv::Mat bezierControlPointsZs;
    cv::Mat bSplineControlPointsZs = grid.getControlPointZs();
    computeBezierControlPointsSimple(bSplineControlPointsZs, bezierControlPointsZs);
    dumpMat("CPU Bezier Control Points", bezierControlPointsZs, 3, 3);
    //gnuPlot3dSurfaceMat("Orig CPU B-Spline Surface", bezierControlPointsZs);

    int nPointsPerDim = 2;
    int nr = bezierControlPointsZs.rows / 3;
    int nc = bezierControlPointsZs.cols / 3;
    cv::Mat evalMatCpu = cv::Mat::zeros(nr * nPointsPerDim, nc * nPointsPerDim, CV_32FC3);
    cv::Rect evalRoi(0, 0, (nc - 1) * nPointsPerDim, (nr - 1) * nPointsPerDim);

    // Canonical impl
    evalBSplineSurfaceCubicPrecomputedMat(bezierControlPointsZs, nPointsPerDim, evalMatCpu);
    dumpMat("CPU Eval Surface", evalMatCpu(evalRoi), nPointsPerDim, nPointsPerDim);
    //printDiffMat(bezierControlPointsZs, bezierControlPointsZsCanonical, 3, 3);

    // CUDA
    int deviceId = 0;
    CudaMat2<float> cudaControlPointZs = CreateCudaMat<float>(bSplineControlPointsZs);
    cv::Mat cudaResultBezierControlPointsZs = cv::Mat::zeros(bezierControlPointsZs.rows, bezierControlPointsZs.cols, CV_32F);
    CudaMat2<float> cudaBezierControlPointZs = CreateCudaMat<float>(cudaResultBezierControlPointsZs);
    cudaComputeBezierControlPoints(deviceId, cudaControlPointZs, cudaBezierControlPointZs);
    //dumpMat("CUDA Bezier Control Points", cudaResultBezierControlPointsZs, 3, 3);
    //printDiffMat(bezierControlPointsZs, cudaResultBezierControlPointsZs, 3, 3);

    cv::Mat evalMatCuda = cv::Mat::zeros(nr * nPointsPerDim, nc * nPointsPerDim, CV_32FC3);
    CudaMat2<CudaPoint3<float>> evalMatCuda2 = CreateCudaMat<CudaPoint3<float>>(evalMatCuda);
    cudaEvalBSplinePrecomp(deviceId, cudaBezierControlPointZs, nPointsPerDim, evalMatCuda2);
    dumpMat("CUDA Eval Surface", evalMatCuda(evalRoi), nPointsPerDim, nPointsPerDim);
    printDiffMat(evalMatCpu(evalRoi), evalMatCuda(evalRoi), nPointsPerDim, nPointsPerDim);

    // Non-precomputed
    cv::Mat evalMatCudaNonPrecomp = cv::Mat::zeros(nr * nPointsPerDim, nc * nPointsPerDim, CV_32FC3);
    CudaMat2<CudaPoint3<float>> evalCudaMat3 = CreateCudaMat<CudaPoint3<float>>(evalMatCudaNonPrecomp);
    CudaMat2<float> bSplineControlPointsZsCudaMat = CreateCudaMat<float>(bSplineControlPointsZs);
    cudaEvalBSpline(deviceId, bSplineControlPointsZsCudaMat, nPointsPerDim, evalCudaMat3);
    dumpMat("CUDA Eval Surface", evalMatCudaNonPrecomp(evalRoi), nPointsPerDim, nPointsPerDim);
    printDiffMat(evalMatCpu(evalRoi), evalMatCudaNonPrecomp(evalRoi), nPointsPerDim, nPointsPerDim);
}

void tryMatStrides()
{
    cv::Mat mat = cv::Mat::zeros(2, 3, CV_32FC3);
    ImageUtil::printMatInfo(mat);
}

void tryCudaTransformImageGray()
{
    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic

    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath, false, pxPerCell);
    saveDebugImage(img, "orig");

    // crop for simple first case
    //img = img(cv::Rect(0, 0, 1280, 1024));

    cv::Mat img8u;
    cv::cvtColor(img, img8u, cv::COLOR_RGB2GRAY);
    saveDebugImage(img8u, "gray");

    ImageTransformBSpline imageTransform(img, pxPerCell);
    cv::Mat dst;

    // random distortion
    imageTransform.setRandomDistortion(-5, 5);

    // CPU
    imageTransform.transformImage(img, interp, dst, false);
    cv::Mat cpuGray;
    cv::cvtColor(dst, cpuGray, cv::COLOR_RGB2GRAY);
    saveDebugImage(cpuGray, "cpuGray");

    //// get bezier control point z's per pixel
    //cv::Mat cpuDebugImage = cv::Mat::zeros(dst.rows, dst.cols, CV_32F);
    //imageTransform.getDxBezierZsPerPixel(cpuDebugImage);
    //saveDebugImage(cpuDebugImage, "cpuDebugImage");

    // CUDA
    int deviceId = 0;

    cv::Mat dxs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dys = imageTransform.getDyGrid().getControlPointZs();
    CudaMat2<float> cudaDxs = CreateCudaMat<float>(dxs);
    CudaMat2<float> cudaDys = CreateCudaMat<float>(dys);

    // 8u
    CudaMat2<uint8_t> srcCuda8u = CreateCudaMat<uint8_t>(img8u);

    cv::Mat dstCuda = cv::Mat::zeros(dst.rows, dst.cols, CV_8U);
    CudaMat2<uint8_t> dstCuda8u = CreateCudaMat<uint8_t>(dstCuda);

    // scale is how the grid relates to the image pixels
    int bSplineGridRows = imageTransform.getDxGrid().rows();
    int bSplineGridCols = imageTransform.getDxGrid().cols();
    float dxScale = (float)dst.cols / bSplineGridCols; // pixels per cell
    float dyScale = (float)dst.rows / bSplineGridRows; // pixels per cell

    cudaBSplineTransformImage(deviceId, srcCuda8u, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCuda8u);
    saveDebugImage(img8u, "gray");
    saveDebugImage(dstCuda, "dstCudaGray");
}

void tryCudaTransformImageBgra()
{
    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic

    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath, false, pxPerCell);
    saveDebugImage(img, "orig");

    ImageTransformBSpline imageTransform(img, pxPerCell);
    cv::Mat dst;

    // random distortion
    imageTransform.setRandomDistortion(-5, 5);

    // CPU
    imageTransform.transformImage(img, interp, dst, false);
    saveDebugImage(dst, "cpu");

    // CUDA
    int deviceId = 0;

    cv::Mat dxs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dys = imageTransform.getDyGrid().getControlPointZs();
    CudaMat2<float> cudaDxs = CreateCudaMat<float>(dxs);
    CudaMat2<float> cudaDys = CreateCudaMat<float>(dys);

    // scale is how the grid relates to the image pixels
    int bSplineGridRows = imageTransform.getDxGrid().rows();
    int bSplineGridCols = imageTransform.getDxGrid().cols();
    float dxScale = (float)dst.cols / bSplineGridCols; // pixels per cell
    float dyScale = (float)dst.rows / bSplineGridRows; // pixels per cell

    // bgra
    cv::Mat imgBgra;
    cv::cvtColor(img, imgBgra, cv::COLOR_BGR2BGRA);
    //saveDebugImage(imgBgra, "imgBgra");
    CudaMat2<BgraQuad> srcCudaBgra = CreateCudaMat<BgraQuad>(imgBgra);
    cv::Mat dstCudaBgra = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC4);
    CudaMat2<BgraQuad> dstCudaBgra2 = CreateCudaMat<BgraQuad>(dstCudaBgra);
    cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);
    //saveDebugImage(imgBgra, "imgBgra");
    saveDebugImage(dstCudaBgra, "dstCudaBgra");
}

void benchTransformImageBgra()
{
    cv::InterpolationFlags interp = cv::INTER_NEAREST; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic

    string imgPath = TestImagePath; // "C:/Temp/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath, false, pxPerCell);
    //saveDebugImage(img, "orig");

    ImageTransformBSpline imageTransform(img, pxPerCell);
    cv::Mat dst = cv::Mat(img.rows, img.cols, CV_8UC3);

    // random distortion
    imageTransform.setRandomDistortion(-5, 5);

    // CPU
    int nCycles;

#if _DEBUG
    nCycles = 1;
#else
    nCycles = 10;
#endif

    auto startTime = CppBaseUtil::getTimeNow();

    for (int i = 0; i < nCycles; i++)
    {
        imageTransform.transformImage(img, interp, dst, false);
    }

    fmt::println("CPU: {0:.1f} ms", CppBaseUtil::getDurationSeconds(startTime) * 1000.0f / nCycles);
    saveDebugImage(dst, "cpu");

    // CUDA
    int deviceId = 0;

    cv::Mat dxs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dys = imageTransform.getDyGrid().getControlPointZs();
    CudaMat2<float> cudaDxs = CreateCudaMat<float>(dxs);
    CudaMat2<float> cudaDys = CreateCudaMat<float>(dys);

    // scale is how the grid relates to the image pixels
    int bSplineGridRows = imageTransform.getDxGrid().rows();
    int bSplineGridCols = imageTransform.getDxGrid().cols();
    float dxScale = (float)dst.cols / bSplineGridCols; // pixels per cell
    float dyScale = (float)dst.rows / bSplineGridRows; // pixels per cell

    // bgra
    cv::Mat imgBgra;
    cv::cvtColor(img, imgBgra, cv::COLOR_BGR2BGRA);
    CudaMat2<BgraQuad> srcCudaBgra = CreateCudaMat<BgraQuad>(imgBgra);
    cv::Mat dstCudaBgra = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC4);
    CudaMat2<BgraQuad> dstCudaBgra2 = CreateCudaMat<BgraQuad>(dstCudaBgra);

    // warmup
    cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);

    auto startTimeCuda = CppBaseUtil::getTimeNow();

    for (int i = 0; i < nCycles; i++)
    {
        cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);
    }

    fmt::println("CUDA: {0:.1f} ms", CppBaseUtil::getDurationSeconds(startTimeCuda) * 1000.0f / nCycles);

    //saveDebugImage(imgBgra, "imgBgra");
    //saveDebugImage(dstCudaBgra, "dstCudaBgra");
}

void runImageTransformCpu(int nCycles, cv::InterpolationFlags interp, ImageTransformBSpline& imageTransform, cv::Mat& img)
{
    cv::Mat dst;

    for (int i = 0; i < nCycles; i++)
    {
        imageTransform.transformImage(img, interp, dst, false);
    }
}

void runImageTransformCuda(int nCycles, cv::InterpolationFlags interp, ImageTransformBSpline& imageTransform, cv::Mat& imgBgra)
{
    int deviceId = 0;
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic

    cv::Mat dxs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dys = imageTransform.getDyGrid().getControlPointZs();
    CudaMat2<float> cudaDxs = CreateCudaMat<float>(dxs);
    CudaMat2<float> cudaDys = CreateCudaMat<float>(dys);

    // scale is how the grid relates to the image pixels
    int bSplineGridRows = imageTransform.getDxGrid().rows();
    int bSplineGridCols = imageTransform.getDxGrid().cols();
    float dxScale = (float)imgBgra.cols / bSplineGridCols; // pixels per cell
    float dyScale = (float)imgBgra.rows / bSplineGridRows; // pixels per cell

    // bgra
    CudaMat2<BgraQuad> srcCudaBgra = CreateCudaMat<BgraQuad>(imgBgra);
    cv::Mat dstCudaBgra = cv::Mat::zeros(imgBgra.rows, imgBgra.cols, CV_8UC4);
    CudaMat2<BgraQuad> dstCudaBgra2 = CreateCudaMat<BgraQuad>(dstCudaBgra);

    for (int i = 0; i < nCycles; i++)
    {
        cudaBSplineTransformImage(deviceId, srcCudaBgra, cudaDxs, dxScale, cudaDys, dyScale, samplingType, dstCudaBgra2);
    }
}

void benchThroughputTransformImageBgra()
{
    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC
    int samplingType = (int)interp; // 0 NN, 1 bilinear, 2 bicubic
    fmt::println("Sampling type: {0}", samplingType);

    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath, false, pxPerCell);
    saveDebugImage(img, "orig");
    fmt::println("RGB Image size: {0} x {1}", img.cols, img.rows);

    ImageTransformBSpline imageTransform(img, pxPerCell);
    cv::Mat dst;

    // random distortion
    imageTransform.setRandomDistortion(-5, 5);

    // CPU
    int nCycles;

#if _DEBUG
    nCycles = 1;
#else
    nCycles = 100;
#endif

    // warmup, and build the ramp images
    //fmt::println("CPU throughput (ms)");
    //fmt::println("nThreads, transformsPerSecond");
    //imageTransform.transformImage(img, interp, dst);

    //for (int nThreads = 1; nThreads <= 24; nThreads++)
    //{
    //    auto startTime = CppBaseUtil::getTimeNow();

    //    // Start N threads each spinning running the transform for nCycles
    //    std::vector<std::thread> threads;

    //    for (int i = 0; i < nThreads; i++)
    //    {
    //        threads.push_back(std::thread(runImageTransformCpu, nCycles, interp, std::ref(imageTransform), std::ref(img)));
    //    }

    //    for (auto& t : threads)
    //    {
    //        t.join();
    //    }

    //    int totalTransformsRun = nThreads * nCycles;
    //    fmt::println("{0}, {1:.1f}", nThreads, totalTransformsRun / CppBaseUtil::getDurationSeconds(startTime));
    //}

    // CUDA
 
    // warmup, and build the ramp images
    fmt::println("CUDA throughput");
    fmt::println("nThreads, transformsPerSecond");
    cv::Mat imgBgra;
    cv::cvtColor(img, imgBgra, cv::COLOR_BGR2BGRA);

    runImageTransformCuda(nCycles, interp, imageTransform, imgBgra);
    fmt::println("Done warmup");

    for (int nThreads = 1; nThreads <= 24; nThreads++)
    {
        auto startTime = CppBaseUtil::getTimeNow();

        // Start N threads each spinning running the transform for nCycles
        std::vector<std::thread> threads;

        for (int i = 0; i < nThreads; i++)
        {
            threads.push_back(std::thread(runImageTransformCuda, nCycles, interp, std::ref(imageTransform), std::ref(imgBgra)));
        }

        for (auto& t : threads)
        {
            t.join();
        }

        int totalTransformsRun = nThreads * nCycles;
        fmt::println("{0}, {1:.1f}", nThreads, totalTransformsRun / CppBaseUtil::getDurationSeconds(startTime));
    }
}

void tryGaussianDomeCurve()
{
    int n = 100;
    vector<cv::Point2f> points;

    for (int i = 0; i < n; i++)
    {
        float x = i;
        float y = 0.0f;
        points.push_back(cv::Point2f(x, y));
    }

    // Add a gaussian
    float halfWidth = 10.0f;
    float height = 5.0f;
    float xCenter = 30.0f;

    for (int i = 0; i < n; i++)
    {
        float x = i;

        points[i].y += gaussianScaledf(halfWidth, height, xCenter, x);
    }

    cv::Mat plotImg = plotPointsAndCurve("Bezier Points", points, points);
    saveDebugImage(plotImg, "bezier-points");
}

void tryGaussianDomeDeform()
{
    bool doDebug = true;
    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC

    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath);
    saveDebugImage(img, "orig");

    int pxPerCell = 32;
    ImageTransformBSpline imageTransform(img, pxPerCell);

    // apply an offset
    float halfWidth = 20.0f;
    cv::Point2f deformCenter(img.cols / 2, img.rows / 2);
    cv::Point2f offset(100.0f, 0.0f);

    applyGaussianDomeDeformation(imageTransform, halfWidth, deformCenter, offset, doDebug);

    cv::Mat dst;
    imageTransform.transformImage(img, interp, dst, doDebug);
    saveDebugImage(dst, "transformed");
}

void applySpringMeshDeformation(cv::Mat& img, ImageTransformBSpline& imageTransform, cv::Point2f deformCenter,
    cv::Point2f offsetVector, bool doDebug)
{
    float pxPerCell = imageTransform.getPxPerCell();
    cv::Mat dxControlMatZs = imageTransform.getDxGrid().getControlPointZs();
    cv::Mat dyControlMatZs = imageTransform.getDyGrid().getControlPointZs();

    // Apply the offset
    int xCenter = lroundf(deformCenter.x / pxPerCell) + 1;
    int yCenter = lroundf(deformCenter.y / pxPerCell) + 1;
    dxControlMatZs.at<float>(yCenter, xCenter) = offsetVector.x;
    dyControlMatZs.at<float>(yCenter, xCenter) = offsetVector.y;

    // Spring mesh
    float k = 0.1f; // spring constant
    float m = 1.0f; // mass
    float friction = 1.0f - 0.015f;

    MatWave dxWave(k, m, friction, dxControlMatZs.rows, dxControlMatZs.cols);
    MatWave dyWave(k, m, friction, dyControlMatZs.rows, dyControlMatZs.cols);
    dxWave.setLockedBorder(true);
    dyWave.setLockedBorder(true);
    cv::Point2i lockedPoint(xCenter, yCenter);
    vector<cv::Point2i> lockedPoints;
    lockedPoints.push_back(lockedPoint);
    dxWave.setLockedPoints(lockedPoints);

    for (int i = 0; i < 150; i++)
    {
        //saveDebugImage(dxControlMatZs, "dxWave");
        dxWave.apply(dxControlMatZs);
        dyWave.apply(dyControlMatZs);

        if (doDebug) // && (i % 10 == 0))
        {
            cv::Mat dst;
            imageTransform.transformImage(img, cv::INTER_CUBIC, dst, false);
            saveDebugImage(dst, "transformed");
        }
    }
}

void trySpringMeshDeform()
{
    bool doDebug = true;
    cv::InterpolationFlags interp = cv::INTER_CUBIC; // cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC

    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath);
    saveDebugImage(img, "orig");

    int pxPerCell = 32;
    ImageTransformBSpline imageTransform(img, pxPerCell);

    // apply an offset
    cv::Point2f deformCenter(img.cols / 2, img.rows / 2);
    cv::Point2f offset(100.0f, 0.0f);

    applySpringMeshDeformation(img, imageTransform, deformCenter, offset, doDebug);

    cv::Mat dst;
    imageTransform.transformImage(img, interp, dst, doDebug);
    saveDebugImage(dst, "transformed");
}


int main()
{
    fmt::print("Starting...\n");

    ImageUtil::init();

#ifndef _DEBUG
    //benchmarkBSplineSurface();
    //tryBSplineGridDeformImage();
    //benchTransformImageBgra();
    //benchThroughputTransformImageBgra();
    showImageTransformBSpline();
    return 0;
#endif

    //tryCvPlot();
    //tryBezierCurve();
    //tryBSplineCurve();
    //tryGnuPlot();
    //tryBezierSurface();
    //tryBSplineSurface();
    //tryComputeBezierControlPoints();
    //try3fMat();
    //tryBezierSurfaceMat();
    //tryBSplineSurfaceMat();
    //tryBSplineSurfaceMat2();
    //tryBSplineGrid();
    //tryMatWave();
    //tryBSplineGridDeformImage();
    //tryMatStrides();
    //tryAllComputeBezierControlPoints();
    //tryCudaEvalBSpline();
    //tryCudaTransformImageGray();
    //tryCudaTransformImageBgra();
    //tryGaussianDomeCurve();
    //tryGaussianDomeDeform();
    //trySpringMeshDeform();

    //tryBezierCurveFit();
    //tryBSplineCurveFit();

    showImageTransformBSpline();

    fmt::print("Done.\n");
    return 0;
}
