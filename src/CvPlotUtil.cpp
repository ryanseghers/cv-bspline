#include <vector>
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

    void cvPlotAddPointSeries(CvPlot::Axes& axes, const vector<cv::Point2f>& points, bool showPoints)
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

    cv::Mat renderPlot(const string& chartTitle, const vector<cv::Point2f>& points, const vector<cv::Point2f>& curvePoints)
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

    cv::Mat plotPoints(const string& chartTitle, const vector<cv::Point2f>& points)
    {
        vector<cv::Point2f> empty;
        return renderPlot(chartTitle, points, empty);
    }

    cv::Mat plotPointsAndCurve(const string& chartTitle, const vector<cv::Point2f>& points, const vector<cv::Point2f>& curvePoints)
    {
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
