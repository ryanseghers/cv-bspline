#include <vector>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "BSplineMiscUtil.h"
#include "ImageUtil.h"

using namespace std;

namespace CvImageDeform
{
    void dumpPoints(const char* title, const std::vector<std::vector<cv::Point3f>>& points)
    {
        fmt::print("---------------------\n");
        fmt::print("{0} ({1}x{2})\n", title, points.size(), points[0].size());

        for (int yi = 0; yi < points.size(); yi++)
        {
            for (int xi = 0; xi < points[yi].size(); xi++)
            {
                fmt::print("({0:.2f}, {1:.2f}, {2:.2f}), ", points[yi][xi].x, points[yi][xi].y, points[yi][xi].z);
            }

            fmt::print("\n");
        }
    }

    void dumpMat(const char* title, const cv::Mat& m)
    {
        fmt::print("---------------------\n");
        fmt::print("{0} ({1}x{2}) {3}\n", title, m.rows, m.cols, CppOpenCVUtil::ImageUtil::getImageTypeString(m.type()));

        for (int yi = 0; yi < m.rows; yi++)
        {
            for (int xi = 0; xi < m.cols; xi++)
            {
                if (m.type() == CV_32FC3)
                {
                    fmt::print("({0:.2f}, {1:.2f}, {2:.2f}), ", m.at<cv::Point3f>(yi, xi).x, m.at<cv::Point3f>(yi, xi).y, m.at<cv::Point3f>(yi, xi).z);
                }
                else if (m.type() == CV_32FC1)
                {
                    fmt::print("{0:.2f}, ", m.at<float>(yi, xi));
                }
            }

            fmt::print("\n");
        }
    }

    std::vector<std::vector<cv::Point3f>> matToPoints(const cv::Mat& mat, cv::Point2f origin, float xyScale)
    {
        std::vector<std::vector<cv::Point3f>> points;

        if (mat.type() == CV_32FC3)
        {
            for (int r = 0; r < mat.rows; r++)
            {
                std::vector<cv::Point3f> row;

                for (int c = 0; c < mat.cols; c++)
                {
                    cv::Point3f p = mat.at<cv::Point3f>(r, c);
                    row.push_back(p);
                }

                points.push_back(row);
            }
        }
        else if (mat.type() == CV_32FC1)
        {
            for (int r = 0; r < mat.rows; r++)
            {
                std::vector<cv::Point3f> row;

                for (int c = 0; c < mat.cols; c++)
                {
                    row.push_back(cv::Point3f(c * xyScale + origin.x, r * xyScale + origin.y, mat.at<float>(r, c)));
                }

                points.push_back(row);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported mat type");
        }

        return points;
    }

    cv::Mat pointsToMatZs(const std::vector<std::vector<cv::Point3f>>& points)
    {
        cv::Mat mat(points.size(), points[0].size(), CV_32FC1);

        for (int r = 0; r < mat.rows; r++)
        {
            for (int c = 0; c < mat.cols; c++)
            {
                mat.at<float>(r, c) = points[r][c].z;
            }
        }

        return mat;
    }

    void setDummyValues(cv::Mat& m)
    {
        for (int r = 0; r < m.rows; r++)
        {
            for (int c = 0; c < m.cols; c++)
            {
                m.at<cv::Point3f>(r, c) = cv::Point3f(c, r, 10.0f);
            }
        }
    }
}
