#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>

#include "CvPlotUtil.h"
#include "ImageUtil.h"
#include <CvPlot/cvplot.h>

using namespace std;

namespace CvImageDeform
{
    void splitPoints(const vector<cv::Point2f>& points, vector<float>& xs, vector<float>& ys)
    {
        for (const auto& p : points)
        {
            xs.push_back(p.x);
            ys.push_back(p.y);
        }
    }

    void cvPlotAddPointSeries(CvPlot::Axes& axes, const vector<cv::Point2f>& points, bool showPoints, std::optional<std::string> lineSpec = std::nullopt)
    {
        vector<float> xs, ys;
        splitPoints(points, xs, ys);

        string finalLineSpec = showPoints ? "-r" : "-b";

        if (lineSpec.has_value())
        {
            finalLineSpec = lineSpec.value();
        }

        CvPlot::Series& series = axes.create<CvPlot::Series>(xs, ys, finalLineSpec);

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

    cv::Mat renderPlot(const string& chartTitle, const vector<cv::Point2f>& points, const vector<cv::Point2f>& curvePoints)
    {
        int drawHeight = 1024;
        int drawWidth = 1024;

        // the image to render to and then display
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

    cv::Mat plotTwoCurves(const std::string& chartTitle, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2)
    {
        int drawHeight = 1024;
        int drawWidth = 1024;

        // the image to render to and then display
        cv::Mat img;
        img.create(drawHeight, drawWidth, CV_8UC3);

        auto axes = makeScatterPlotAxes(chartTitle);

        if (!points1.empty())
        {
            cvPlotAddPointSeries(axes, points1, false, "-g");
        }

        if (!points2.empty())
        {
            cvPlotAddPointSeries(axes, points2, false);
        }

        img = axes.render(drawHeight, drawWidth);
        return img;
    }

    cv::Mat plotPoints(const string& chartTitle, const vector<cv::Point2f>& points)
    {
        vector<cv::Point2f> empty;
        return renderPlot(chartTitle, points, empty);
    }

    cv::Mat plotPointsAndCurve(const string& chartTitle, const vector<cv::Point2f>& points, const vector<cv::Point2f>& curvePoints)
    {
        return renderPlot(chartTitle, points, curvePoints);
    }

    // For uniform points from 0 to n-1
    cv::Mat plotPointsAndCurve(const string& chartTitle, const vector<float>& pointValues, const vector<cv::Point2f>& curvePoints)
    {
        vector<cv::Point2f> points;

        for (int i = 0; i < pointValues.size(); i++)
        {
            points.push_back(cv::Point2f(i, pointValues[i]));
        }

        return renderPlot(chartTitle, points, curvePoints);
    }

    void tryCvPlot()
    {
        // create a cvplot
        vector<float> xs = { 0, 1, 2, 3, 4, 5 };
        vector<float> ys = { 0, 1, 2, 1, 3, 1 };
        vector<cv::Point2f> points;
        for (int i = 0; i < xs.size(); i++)
        {
            points.push_back(cv::Point2f(xs[i], ys[i]));
        }
        vector<cv::Point2f> empty;
        cv::Mat plotImg = renderPlot("Test Plot", points, empty);
        CppOpenCVUtil::saveDebugImage(plotImg, "plot");
    }
}
