#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MatWave.h"
#include "ImageUtil.h"

using namespace CppOpenCVUtil;


namespace CvImageDeform
{
    MatWave::MatWave(float kSpring, float mass, float friction, int rows, int cols)
    {
        this->kSpring = kSpring;
        this->mass = mass;
        this->friction = friction;
        velocity = cv::Mat::zeros(rows, cols, CV_32F);
        mask = cv::Mat::ones(rows, cols, CV_32F);
        borderLockMask = cv::Mat::ones(rows, cols, CV_32F);
        storedLockedPointMask = cv::Mat::ones(rows, cols, CV_32F);
    }

    void MatWave::setLockedBorder(int borderWidth)
    {
        lockedBorderWidth = borderWidth;

        borderLockMask = mask.clone();

        // set border to 0 (by setting all to 0 and then center to 1)
        borderLockMask = 0.0f;
        cv::Rect roi(lockedBorderWidth, lockedBorderWidth, mask.cols - lockedBorderWidth * 2, mask.rows - lockedBorderWidth * 2);
        borderLockMask(roi) = 1.0f;
    }

    void MatWave::setLockedPoints(const std::vector<cv::Point2i>& points)
    {
        for (int i = 0; i < points.size(); i++)
        {
            mask.at<float>(points[i].y, points[i].x) = 0.0f;
        }
    }

    void MatWave::setLockedPoint(int x, int y)
    {
        mask.at<float>(y, x) = 0.0f;
    }

    void MatWave::storeLockedPoints()
    {
        cv::min(storedLockedPointMask, mask, storedLockedPointMask);
    }

    void MatWave::clearLockedPoints()
    {
        // if we have a locked border have to re-apply else we lose that
        mask = 1.0f;
    }

    void MatWave::clearStoredLockedPoints()
    {
        // if we have a locked border have to re-apply else we lose that
        storedLockedPointMask = 1.0f;
    }

    void MatWave::saveDebugImages(const char* name)
    {
        saveDebugImage(mask, name);
        saveDebugImage(force, "force");
        saveDebugImage(acceleration, "acceleration");
        saveDebugImage(velocity, "velocity");
        saveDebugImage(mask, "lockedPointsMask");
        saveDebugImage(storedLockedPointMask, "storedLockedPointMask");
    }

    void MatWave::apply(cv::Mat& mat)
    {
        // Force is the sum of the f=k*x of the 4 neighbors of each point.
        force = cv::Mat::zeros(mat.rows, mat.cols, CV_32F);

        for (int r = 0; r < mat.rows; r++)
        {
            for (int c = 0; c < mat.cols; c++)
            {
                float z = mat.at<float>(r, c);
                float dz = 0.0f;

                if (r > 0)
                {
                    dz += mat.at<float>(r - 1, c) - z;
                }

                if (r < mat.rows - 1)
                {
                    dz += mat.at<float>(r + 1, c) - z;
                }

                if (c > 0)
                {
                    dz += mat.at<float>(r, c - 1) - z;
                }

                if (c < mat.cols - 1)
                {
                    dz += mat.at<float>(r, c + 1) - z;
                }

                force.at<float>(r, c) = kSpring * dz;
            }
        }

        // Acceleration is the force over mass.
        acceleration = force / mass;

        // Apply the acceleration to the velocity.
        float dt = 1.0f;
        velocity += acceleration * dt;

        // Mask the velocity so we don't change the locked points
        //saveDebugImage(velocity, "velocity");
        //saveDebugImage(mask, "mask");
        cv::multiply(velocity, mask, velocity);
        cv::multiply(velocity, borderLockMask, velocity);
        cv::multiply(velocity, storedLockedPointMask, velocity);

        // Apply the velocity to the position.
        mat += velocity * dt;

        // Apply friction to the velocity.
        velocity *= friction;

        // Truncate low velocity to zero
        float minVelocity = 0.0001f;

        for (int r = 0; r < velocity.rows; r++)
        {
            for (int c = 0; c < velocity.cols; c++)
            {
                if (std::abs(velocity.at<float>(r, c)) < minVelocity)
                {
                    velocity.at<float>(r, c) = 0.0f;
                }
            }
        }
    }
}
