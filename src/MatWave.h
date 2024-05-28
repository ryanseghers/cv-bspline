#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    /**
     * @brief A grid where each point has springs to its 4 neighbors.
     */
    class MatWave
    {
    private:
        float kSpring;
        float mass;
        float friction; // 1.0 is no friction
        cv::Mat velocity;

        int lockedBorderWidth = 0;

        /**
         * @brief Locked points are locked by a mask that is multiplied by the
         * velocity at each round, before applying the velocity.
         */
        cv::Mat mask;

        /**
         * @brief Last locked point so we can clear it.
         */
        cv::Point2i lastLockedPoint;

        MatWave() = delete;

    public:
        /**
         * @brief Ctor.
         * @param kSprint Sprint constant.
         * @param mass Mass of each point.
        */
        MatWave(float kSpring, float mass, float friction, int rows, int cols);

        /**
         * @brief This doesn't clear borders from previous calls so you cannot reduce the border
         * width.
         */
        void setLockedBorder(int /*borderWidth*/);

        void setLockedPoints(const std::vector<cv::Point2i>& points);
        void setLockedPoint(int x, int y);


        /**
         * @brief In-place apply one tick's worth of wave propagation.
        */
        void apply(cv::Mat& mat);
    };
}
