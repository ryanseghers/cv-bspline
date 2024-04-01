#pragma once
#include "cuda_runtime.h"
#include "cudaUtil.h"


//cudaError_t cudaBSplineTransformImage(const CudaMat2<uint8_t>& inputImage, const CudaMat2<float>& dxControlPointZs, const CudaMat2<float>& dyControlPointZs, const CudaMat2<uint8_t>& outputImage);

bool cudaComputeBezierControlPoints(int deviceId, CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs);
bool cudaEvalBSpline(int deviceId, CudaMat2<float>& controlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat);
