#include "BSplineGrid.h"
#include "ImageTransformBSpline.h"
#include "ImageUtil.h"

using namespace std;
using namespace CppOpenCVUtil;

namespace CvImageDeform
{
    ImageTransformBSpline::ImageTransformBSpline(const cv::Mat& image, int pxPerCell)
        : dxGrid(image.rows / pxPerCell, image.cols / pxPerCell),
        dyGrid(image.rows / pxPerCell, image.cols / pxPerCell)
    {
    }

    void ImageTransformBSpline::transformImage(const cv::Mat& inputImage, cv::InterpolationFlags interp, cv::Mat& outputImage)
    {
        int rowRes = inputImage.rows / dxGrid.rows();
        int colRes = inputImage.cols / dxGrid.cols();

        // remap takes x,y coords, but we have dx,dy, so we need some mats with x,y coords to add our dx,dy to.
        if (ImageUtil::ensureMat(xcoords, inputImage.rows, inputImage.cols, CV_32FC1))
        {
            ImageUtil::setToColIndices<float>(xcoords);
        }

        if (ImageUtil::ensureMat(ycoords, inputImage.rows, inputImage.cols, CV_32FC1))
        {
            ImageUtil::setToRowIndices<float>(ycoords);
        }

        // eval to get dx,dy for every pixel in image
        cv::Mat dxEvalMat = dxGrid.evalSurface(colRes);
        cv::Mat dyEvalMat = dyGrid.evalSurface(rowRes);

        // eval mats are 3 channel images, but we only need the Z channel.
        std::vector<cv::Mat> channels(3);
        cv::split(dxEvalMat, channels);
        cv::Mat dxEvalMatZs = channels[2];
        
        cv::split(dyEvalMat, channels);
        cv::Mat dyEvalMatZs = channels[2];

        // remap takes x coordinate to sample
        cv::Mat xmap = xcoords + dxEvalMatZs;
        cv::Mat ymap = ycoords + dyEvalMatZs;

        cv::Mat dest;
        cv::remap(inputImage, outputImage, xmap, ymap, interp, cv::BORDER_CONSTANT);
    }

    void ImageTransformBSpline::setRandomDistortion(float min, float max)
    {
        dxGrid.fillRandomControlPoints(min, max);
        dyGrid.fillRandomControlPoints(min, max);
    }
}
