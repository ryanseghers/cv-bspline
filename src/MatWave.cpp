#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MatWave.h"

namespace CvImageDeform
{
    MatWave::MatWave(float kSpring, float mass, float friction, int rows, int cols)
    {
        this->kSpring = kSpring;
        this->mass = mass;
        this->friction = friction;
        velocity = cv::Mat::zeros(rows, cols, CV_32F);
    }

    void MatWave::apply(cv::Mat& mat)
    {
        // Force is the sum of the f=k*x of the 4 neighbors of each point.
        cv::Mat force = cv::Mat::zeros(mat.rows, mat.cols, CV_32F);

        for (int r = 0; r < mat.rows; r++)
        {
            for (int c = 0; c < mat.cols; c++)
            {
                float z = mat.at<float>(r, c);
                float df = 0.0f;

                if (r > 0)
                {
                    df += mat.at<float>(r - 1, c) - z;
                }

                if (r < mat.rows - 1)
                {
                    df += mat.at<float>(r + 1, c) - z;
                }

                if (c > 0)
                {
                    df += mat.at<float>(r, c - 1) - z;
                }

                if (c < mat.cols - 1)
                {
                    df += mat.at<float>(r, c + 1) - z;
                }

                force.at<float>(r, c) = kSpring * df;
            }
        }

        // Acceleration is the force over mass.
        cv::Mat acceleration = force / mass;

        // Apply the acceleration to the velocity.
        float dt = 1.0f;
        velocity += acceleration * dt;

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
