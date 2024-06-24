#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    /**
     * @brief A grid where each point has springs to its 4 neighbors.
     * This just keeps track of velocity, updates with each call to apply,
     * and add that velocity 
     */
    class MatWave
    {
    private:
        float kSpring;
        float mass;
        float friction; // 1.0 is no friction
        cv::Mat velocity;
        cv::Mat force;
        cv::Mat acceleration;
        cv::Mat borderLockMask;

        int lockedBorderWidth = 0;

        /**
         * @brief Locked points are locked by a mask that is multiplied by the
         * velocity at each round, before applying the velocity.
         */
        cv::Mat mask;
        cv::Mat storedLockedPointMask;

        MatWave() = delete;

    public:
        /**
         * @brief Ctor.
         * @param kSprint Sprint constant.
         * @param mass Mass of each point.
        */
        MatWave(float kSpring, float mass, float friction, int rows, int cols);

        /**
         * @brief This clears all previous locked points.
         */
        void setLockedBorder(int borderWidth);

        void setLockedPoints(const std::vector<cv::Point2i>& points);
        void setLockedPoint(int x, int y);
        void clearLockedPoints();

        /**
        * @brief Store current locked points and keep applying them until clear.
        */
        void storeLockedPoints();
        void clearStoredLockedPoints();

        /**
         * @brief In-place apply one tick's worth of wave propagation.
        */
        void apply(cv::Mat& mat);

        void saveDebugImages(const char* name);
    };
}
