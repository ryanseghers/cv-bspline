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

    void dumpMat(const char* title, const cv::Mat& m, int cellWidth, int cellHeight)
    {
        fmt::print("---------------------\n");
        fmt::print("{0} ({1}x{2}) {3}\n", title, m.rows, m.cols, CppOpenCVUtil::ImageUtil::getImageTypeString(m.type()));

        for (int yi = 0; yi < m.rows; yi++)
        {
            if ((cellHeight > 0) && (yi % cellHeight == 0))
            {
                fmt::print("\n");
            }

            for (int xi = 0; xi < m.cols; xi++)
            {
                if ((cellWidth > 0) && (xi % cellWidth == 0))
                {
                    fmt::print(" ");
                }

                if (m.type() == CV_32FC3)
                {
                    auto pt = m.at<cv::Point3f>(yi, xi);
                    fmt::print("({0:.2f}, {1:.2f}, {2:.2f}), ", pt.x, pt.y, pt.z);
                }
                else if (m.type() == CV_32FC1)
                {
                    fmt::print("{0:.2f}, ", m.at<float>(yi, xi));
                }
            }

            fmt::print("\n");
        }
    }

    void printDiffMat(const cv::Mat& mat1, const cv::Mat& mat2, int cellWidth, int cellHeight)
    {
        fmt::println("Deltas:");

        if (mat1.size() != mat2.size())
        {
            fmt::print("Size mismatch\n");
            return;
        }

        if (mat1.type() != mat2.type())
        {
            fmt::print("Type mismatch\n");
            return;
        }

        if (mat1.type() == CV_32F)
        {
            for (int r = 0; r < mat1.rows; ++r)
            {
                if ((cellHeight > 0) && (r % cellHeight == 0))
                {
                    fmt::print("\n");
                }

                for (int c = 0; c < mat1.cols; ++c)
                {
                    if ((cellWidth > 0) && (c % cellWidth == 0))
                    {
                        fmt::print(" ");
                    }

                    float delta = abs(mat1.at<float>(r, c) - mat2.at<float>(r, c));

                    if (delta > 0.0001f)
                    {
                        fmt::print("{0:.2f}, ", delta);
                    }
                    else
                    {
                        fmt::print("----, ");
                    }
                }

                fmt::print("\n");
            }
        }
        else
        {
            cv::Mat diffMat = cv::abs(mat1 - mat2);
            dumpMat("Diff", diffMat);
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
