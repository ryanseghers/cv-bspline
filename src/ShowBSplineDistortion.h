#include <string>

namespace CvImageDeform
{
    void trySpringMeshDeform();
    void tryGaussianDomeDeform();
    void tryGaussianDomeCurve();
    void benchThroughputTransformImageBgra();
    void tryCudaTransformImageBgra();
    void showImageTransformBSpline(const std::string& testImagePath);
}
