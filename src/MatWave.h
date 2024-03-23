#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    class MatWave
    {
    private:
        float kSpring;
        float mass;
        float friction; // 1.0 is no friction
        cv::Mat velocity;

        MatWave() = delete;

    public:
        /**
         * @brief Ctor.
         * @param kSprint Sprint constant.
         * @param mass Mass of each point.
        */
        MatWave(float kSpring, float mass, float friction, int rows, int cols);

        /**
         * @brief In-place apply one tick's worth of wave propagation.
        */
        void apply(cv::Mat& mat);
    };
}
