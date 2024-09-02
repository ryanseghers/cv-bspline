#include "BezierMat3d.h"
#include "BezierUtil.h"

using namespace std;

namespace CvImageDeform
{
    float evalBezierVolumeCubicPointMat3d(const cv::Mat& controlPointsZs, float u, float v, float w)
    {
        float value = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    float cp = controlPointsZs.at<float>(i, j, k);
                    value += bezierPolyTerm(i, u) * bezierPolyTerm(j, v) * bezierPolyTerm(k, w) * cp;
                }
            }
        }

        return value;
    }
}
