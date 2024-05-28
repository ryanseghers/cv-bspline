#pragma once
#include "BSplineGrid.h"
#include <opencv2/opencv.hpp>

namespace CvImageDeform
{
    /**
     * @brief This defines a deformation field over an image using B-Splines.
     * The x and y components of the deformation field are defined by separate B-Spline grids.
     */
    class ImageTransformBSpline
    {
    private:
        BSplineGrid dxGrid;
        BSplineGrid dyGrid;

        float pxPerCell;

        // Pre-computed x and y coordinates for remap.
        cv::Mat xcoords;
        cv::Mat ycoords;

    public:
        /**
         * @brief ctor.
         * @param image Just for the dims. 
         * @param pxPerCell Specify the resolution of the grid.
         */
        ImageTransformBSpline(const cv::Mat& image, int pxPerCell);

        BSplineGrid& getDxGrid() { return dxGrid; }
        BSplineGrid& getDyGrid() { return dyGrid; }
        float getPxPerCell() { return pxPerCell; }

        void transformImage(const cv::Mat& inputImage, cv::InterpolationFlags interp, cv::Mat& outputImage, bool doDebug);

        void setRandomDistortion(float min, float max);
    };
}
