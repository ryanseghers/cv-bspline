#include <string>
#include "mainMiscUtil.h"

using namespace std;


/**
 * @brief Eval gaussian with mean 0 and sigma 1.
 */
float gaussianf(float x)
{
    float sigma = 1.0f; // Standard deviation
    return std::expf(-0.5f * std::powf(x / sigma, 2));
}

/**
* @brief Using the gaussian function in a geometric sense, just for its shape.
* @param halfWidth Half width of peak.
* @param height Max height of peak.
* @param center X center of peak.
* @param x Evaluate the function at this x.
*/
float gaussianScaledf(float halfWidth, float height, float center, float x)
{
    float xOffset = x - center;
    float xScaled = xOffset / (halfWidth / 2);
    float g = gaussianf(xScaled) * height;
    return g;
}

cv::Mat ensureImageDims(cv::Mat img, int sizeMult)
{
    if ((img.rows % sizeMult) || (img.cols % sizeMult))
    {
        cv::Mat tmp;
        int newRows = (img.rows / sizeMult) * sizeMult;
        int newCols = (img.cols / sizeMult) * sizeMult;
        cv::resize(img, tmp, cv::Size(newCols, newRows));
        img = tmp;
    }

    return img;
}

cv::Mat loadAndConvertTestImage(const string& imgPath, int screenWidth, int screenHeight, bool doConvertTo8u, int sizeMult)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    cv::Mat tmp;
    //saveDebugImage(img, "orig");

    if (doConvertTo8u && (img.type() != CV_8U))
    {
        cv::cvtColor(img, tmp, cv::COLOR_RGB2GRAY);
        img = tmp;
        //saveDebugImage(img, "gray");
    }

    // Downsize for perf
    if (img.rows > screenHeight)
    {
        int newRows = screenHeight;
        int newCols = screenWidth * img.cols / img.rows;
        cv::resize(img, img, cv::Size(newCols, newRows));
        //saveDebugImage(img, "scaled");
    }

    // Make image size a multiple of sizeMult for simplicity
    img = ensureImageDims(img, sizeMult);

    return img;
}
