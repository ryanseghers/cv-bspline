#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace std;
using namespace cv;

/**
 * @brief Compute the knot vector for a clamped B-spline.
 * @param numControlPoints - Number of control points.
 * @param degree - Degree of the B-spline.
 * @param knotVector - The computed knot vector (output).
 */
void computeKnotVector(int numControlPoints, int degree, vector<float>& knotVector)
{
    int n = numControlPoints - 1; // Last control point index
    int m = n + degree + 1;       // Last knot index

    knotVector.resize(m + 1);

    // Clamped knot vector with multiplicity degree + 1 at the ends
    for (int i = 0; i <= m; ++i)
    {
        if (i <= degree)
            knotVector[i] = 0.0f;
        else if (i >= m - degree)
            knotVector[i] = 1.0f;
        else
            knotVector[i] = (float)(i - degree) / (m - 2 * degree);
    }
}

/**
 * @brief Find the knot span index for a given parameter u.
 * @param n - Number of control points minus 1.
 * @param degree - Degree of the B-spline.
 * @param u - Parameter value.
 * @param knotVector - Knot vector.
 * @return Knot span index.
 */
int findKnotSpan(int n, int degree, float u, const vector<float>& knotVector)
{
    if (u >= knotVector[n + 1])
        return n;

    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;

    while (u < knotVector[mid] || u >= knotVector[mid + 1])
    {
        if (u < knotVector[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }

    return mid;
}

/**
 * @brief Compute the non-zero B-spline basis functions at u.
 * @param span - Knot span index.
 * @param u - Parameter value.
 * @param degree - Degree of the B-spline.
 * @param knotVector - Knot vector.
 * @param N - Basis functions (output).
 */
void computeBasisFunctions(int span, float u, int degree, const vector<float>& knotVector, vector<float>& N)
{
    N[0] = 1.0f;
    vector<float> left(degree + 1);
    vector<float> right(degree + 1);

    for (int j = 1; j <= degree; ++j)
    {
        left[j] = u - knotVector[span + 1 - j];
        right[j] = knotVector[span + j] - u;
        float saved = 0.0f;

        for (int r = 0; r < j; ++r)
        {
            float temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        N[j] = saved;
    }
}

/**
 * @brief Fit a cubic B-spline curve to a set of target points.
 * The number of control points is equal to the number of target points.
 * @param targetPoints - The points to fit to.
 * @param controlPointsMat - Control points (output). This needs to already be allocated with 1 column and the same number of rows as targetPoints.
 */
// This is not quite correct. The input points are uniform (x values 0 to N-1 with step 1), but the output
// control points, when evaluated at those X values, give a curve that is squished.
// I suspect this function expands the X range to handle edges, then doesn't undo that expansion.
void fitBSplineCurveBasisLinAlg(vector<cv::Point2f>& targetPoints, cv::Mat& controlPointsMat)
{
    int n = static_cast<int>(targetPoints.size()) - 1; // Last point index
    int degree = 3;

    if (n < degree)
    {
        throw std::invalid_argument("Number of target points must be at least degree + 1.");
    }

    // Number of control points is equal to number of target points
    int numControlPoints = n + 1;

    // Compute knot vector
    vector<float> knotVector;
    computeKnotVector(numControlPoints, degree, knotVector);

    // Parameter values u_i (using chord length parameterization)
    vector<float> u(n + 1, 0.0f);
    float totalLength = 0.0f;
    for (int i = 1; i <= n; ++i)
    {
        totalLength += norm(targetPoints[i] - targetPoints[i - 1]);
    }
    if (totalLength == 0.0f)
    {
        throw std::runtime_error("Total length of target points is zero.");
    }
    vector<float> chordLengths(n + 1, 0.0f);
    float cumulativeLength = 0.0f;
    for (int i = 1; i <= n; ++i)
    {
        cumulativeLength += norm(targetPoints[i] - targetPoints[i - 1]);
        u[i] = cumulativeLength / totalLength;
    }

    // Assemble the basis function matrix N
    Mat Nmat = Mat::zeros(n + 1, numControlPoints, CV_32F);

    vector<float> N(degree + 1);
    for (int i = 0; i <= n; ++i)
    {
        float u_i = u[i];
        int span = findKnotSpan(n, degree, u_i, knotVector);
        computeBasisFunctions(span, u_i, degree, knotVector, N);

        for (int j = 0; j <= degree; ++j)
        {
            int col = span - degree + j;
            Nmat.at<float>(i, col) = N[j];
        }
    }

    // Prepare the y-values of target points
    Mat P_y(n + 1, 1, CV_32F);
    for (int i = 0; i <= n; ++i)
    {
        P_y.at<float>(i, 0) = targetPoints[i].y;
    }

    // Solve for control points y-values
    Mat NtN = Nmat.t() * Nmat;
    Mat NtP_y = Nmat.t() * P_y;

    Mat C_y;

    // Solve NtN * C_y = NtP_y
    bool success_y = solve(NtN, NtP_y, C_y, DECOMP_SVD);

    if (!success_y)
    {
        throw std::runtime_error("Failed to solve for control points.");
    }

    // Store the y-values of control points in controlPointsMat
    for (int i = 0; i < numControlPoints; ++i)
    {
        controlPointsMat.at<float>(i, 0) = C_y.at<float>(i, 0);
    }
}
