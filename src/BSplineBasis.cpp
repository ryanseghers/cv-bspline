#include <opencv2/opencv.hpp>
#include <cmath>

// direct B-spline evaluation using the basis function or B-spline basis evaluation

namespace CvImageDeform
{
    // Function to evaluate the cubic B-spline basis functions
    void computeCubicBSplineWeights(float t, float* weights)
    {
        float t2 = t * t;
        float t3 = t2 * t;

        weights[0] = ((1 - t) * (1 - t) * (1 - t)) / 6.0f;
        weights[1] = (3 * t3 - 6 * t2 + 4) / 6.0f;
        weights[2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0f;
        weights[3] = t3 / 6.0f;
    }

    /**
    * Evaluate the B-spline at a given 1D point
    * @param coeffs Control points (coefficients) in a 1d Mat with coefficients in columns: cv::Mat m(9, 1, CV_32F)
    * @param x The point at which to evaluate the B-spline
    */
    float evaluateBSpline1d(const cv::Mat& coeffs, float x)
    {
        // Get the dimensions of the coeffs
        int size = coeffs.size[0];

        if (size < 4)
        {
            throw std::invalid_argument("Volume must have at least 4 elements (in mat rows) for cubic B-spline evaluation.");
        }

        int ix = static_cast<int>(std::floor(x));
        float tx = x - ix;

        // Compute the cubic B-spline weights for each dimension
        float wx[4];
        computeCubicBSplineWeights(tx, wx);

        // Loop over the influencing control points (4 in each dimension)
        float result = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            int xIndex = ix - 1 + i;
            if (xIndex < 0 || xIndex >= size) continue;
            float coeff = coeffs.at<float>(xIndex);
            result += coeff * wx[i];
        }

        return result;
    }

    // Function to evaluate the B-spline at a given 2D point
    float evaluateBSpline2d(const cv::Mat& coeffs, float x, float y)
    {
        // Get the dimensions of the coeffs
        int sizes[2] = { coeffs.size[0], coeffs.size[1] };

        // Floor of x, y, z to get the base indices
        int ix = static_cast<int>(std::floor(x));
        int iy = static_cast<int>(std::floor(y));

        // Fractional parts
        float tx = x - ix;
        float ty = y - iy;

        // Compute the cubic B-spline weights for each dimension
        float wx[4], wy[4];
        computeCubicBSplineWeights(tx, wx);
        computeCubicBSplineWeights(ty, wy);

        // Loop over the influencing control points (4 in each dimension)
        float result = 0.0f;

        for (int j = 0; j < 4; ++j)
        {
            int yIndex = iy - 1 + j;
            if (yIndex < 0 || yIndex >= sizes[1]) continue;

            for (int i = 0; i < 4; ++i)
            {
                int xIndex = ix - 1 + i;
                if (xIndex < 0 || xIndex >= sizes[0]) continue;

                int idx[2] = { xIndex, yIndex };
                float coeff = coeffs.at<float>(idx);

                result += coeff * wx[i] * wy[j];
            }
        }

        return result;
    }

    // Function to evaluate the B-spline at a given 3D point
    float evaluateBSpline3d(const cv::Mat& volume, float x, float y, float z)
    {
        // Get the dimensions of the volume
        int sizes[3] = { volume.size[0], volume.size[1], volume.size[2] };

        // Floor of x, y, z to get the base indices
        int ix = static_cast<int>(std::floor(x));
        int iy = static_cast<int>(std::floor(y));
        int iz = static_cast<int>(std::floor(z));

        // Fractional parts
        float tx = x - ix;
        float ty = y - iy;
        float tz = z - iz;

        // Compute the cubic B-spline weights for each dimension
        float wx[4], wy[4], wz[4];
        computeCubicBSplineWeights(tx, wx);
        computeCubicBSplineWeights(ty, wy);
        computeCubicBSplineWeights(tz, wz);

        // Loop over the influencing control points (4 in each dimension)
        float result = 0.0f;

        for (int k = 0; k < 4; ++k)
        {
            int zIndex = iz - 1 + k;
            if (zIndex < 0 || zIndex >= sizes[2]) continue;

            for (int j = 0; j < 4; ++j)
            {
                int yIndex = iy - 1 + j;
                if (yIndex < 0 || yIndex >= sizes[1]) continue;

                for (int i = 0; i < 4; ++i)
                {
                    int xIndex = ix - 1 + i;
                    if (xIndex < 0 || xIndex >= sizes[0]) continue;

                    int idx[3] = { xIndex, yIndex, zIndex };
                    float coeff = volume.at<float>(idx);

                    result += coeff * wx[i] * wy[j] * wz[k];
                }
            }
        }

        return result;
    }
}
