#include <string>
#include <chrono>
#include <ctime>
#include <future>
#include <optional>
#include <fstream>
#include <iostream>
#include <sstream>

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
#include "mainMiscUtil.h"

#ifdef _WIN32
#include "ShowBSplineDistortion.h"
#include "cudaUtil.h"
#include "cudaBSplineCoeffs.h"
#include "cudaBSplineEval.h"
#include "cudaBSplineTransform.h"
#endif

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

//     // simd
//     pt = evalBezierSurfaceCubicPointMatSimd(controlMatZs, 0.1f, 0.2f);
//     fmt::print("Point: {0}, {1}, {2}\n", pt.x, pt.y, pt.z);

//     // simd
//     pt = evalBezierSurfaceCubicPointMatSimd2(controlMatZs, 0.1f, 0.2f);
//     fmt::print("Point: {0}, {1}, {2}\n", pt.x, pt.y, pt.z);
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
    //evalBezierSurfaceCubicMatAvx(controlMatZs, 10, 0.0f, 0.0f, evalMat);
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
    cv::Mat img = loadAndConvertTestImage(imgPath, screenWidth, screenHeight);

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

std::vector<float> loadTransformParameters(const std::string& filename)
{
    std::ifstream file(filename);
    std::vector<float> parameters;
    std::string line;

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return parameters;
    }

    while (std::getline(file, line))
    {
        size_t idx = line.find("TransformParameters");

        if ((idx != std::string::npos) && (idx < 3))
        {
            string restOfLine = line.substr(idx + 19);
            std::istringstream iss(restOfLine);

            float value;
            while (iss >> value)
            {
                parameters.push_back(value);
            }

            break;
        }
    }

    file.close();
    return parameters;
}

void tryLoadBSplineParams()
{
#ifdef _WIN32
    string path = "C:/Projects/2024-07-12-deformable-registration/TG_132_Test7/TransformParameters.1.txt";
#else
    string path = "/Users/ryanseghers/tmp/TransformParameters.1.txt";
#endif
    vector<float> params = loadTransformParameters(path);
    fmt::print("Params: {}\n", params.size());
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
#ifdef _WIN32
    showImageTransformBSpline(TestImagePath);
    return 0;
#endif
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

    //showImageTransformBSpline(TestImagePath);
    tryLoadBSplineParams();

    fmt::print("Done.\n");
    return 0;
}
