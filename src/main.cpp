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

using namespace std;
using namespace std::chrono;
using namespace CppOpenCVUtil;
using namespace CvImageDeform;

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
    computeBezierControlPoints(zs, bezierControlMatZs);

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
    if (img.rows > 1024)
    {
        int newRows = 1024;
        int newCols = 1024 * img.cols / img.rows;
        cv::resize(img, img, cv::Size(newCols, newRows));
        //saveDebugImage(img, "scaled");
    }

    // Make image size a multiple of sizeMult for simplicity
    if ((img.rows % sizeMult) || (img.cols % sizeMult))
    {
        int newRows = (img.rows / sizeMult) * sizeMult;
        int newCols = (img.cols / sizeMult) * sizeMult;
        cv::resize(img, tmp, cv::Size(newCols, newRows));
        img = tmp;
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
        computeBezierControlPoints(zs, bezierControlPointsZs);
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

int pxPerCell = 32;
cv::Mat dxControlMatZs;
cv::Mat dyControlMatZs;

void mouseCallback(int event, int x, int y, int flags, void* userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN) 
    {
        std::cout << "Left button clicked at (" << x << ", " << y << ")" << std::endl;

        // control points are outside of the surface
        int xi = x / pxPerCell;
        int yi = y / pxPerCell;
        dxControlMatZs.at<float>(yi + 1, xi + 1) = 50.0f;
        dyControlMatZs.at<float>(yi + 1, xi + 1) = 50.0f;
    }
}

void tryImageTransformBSpline()
{
    string imgPath = "Z:/Camera/Pictures/2023/2023-09-14 Dan Marmot Lake/PXL_20230914_161024366.jpg";
    cv::Mat img = loadAndConvertTestImage(imgPath, false, pxPerCell);

    ImageTransformBSpline imageTransform(img, pxPerCell);
    cv::Mat dst;
    //imageTransform.transformImage(img, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "dst");

    //// random distortion
    //imageTransform.setRandomDistortion(-10, 10);
    //imageTransform.transformImage(img, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "dst");

    // wave
    dxControlMatZs = imageTransform.getDxGrid().getControlPointZs();
    dyControlMatZs = imageTransform.getDyGrid().getControlPointZs();

    float k = 0.1f; // spring constant
    float m = 1.0f; // mass
    float friction = 1.0f - 0.015f;

    MatWave dxWave(k, m, friction, dxControlMatZs.rows, dxControlMatZs.cols);
    MatWave dyWave(k, m, friction, dyControlMatZs.rows, dyControlMatZs.cols);
    int nCycles = 100;

    // initial perturbation
    int xCenter = dxControlMatZs.cols / 2;
    int yCenter = dxControlMatZs.rows / 2;
    //dxControlMatZs.at<float>(yCenter, xCenter) = 50.0f; // initial wave
    //dyControlMatZs.at<float>(yCenter, xCenter) = 50.0f; // initial wave

    //// save debug images 
    //for (int i = 0; i < nCycles; ++i)
    //{
    //    fmt::println("Step {}", i);

    //    dxWave.apply(dxControlMatZs);
    //    dyWave.apply(dyControlMatZs);

    //    imageTransform.transformImage(img, cv::INTER_CUBIC, dst);
    //    saveDebugImage(dst, "dst");
    //}

    // interactive window
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Image", mouseCallback);

    while(true)
    {
        dxWave.apply(dxControlMatZs);
        dyWave.apply(dyControlMatZs);

        imageTransform.transformImage(img, cv::INTER_LINEAR, dst);

        cv::imshow("Image", dst);
        cv::waitKey(10);
    }

    cv::destroyAllWindows();
}

int main()
{
    fmt::print("Starting...\n");

    ImageUtil::init();

#ifndef _DEBUG
    //benchmarkBSplineSurface();
    //tryBSplineGridDeformImage();
    tryImageTransformBSpline();
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
    tryImageTransformBSpline();

    fmt::print("Done.\n");
    return 0;
}
