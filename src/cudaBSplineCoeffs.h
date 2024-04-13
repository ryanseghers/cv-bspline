#pragma once
#include "cuda_runtime.h"
#include "cudaUtil.h"


extern __global__ void computeBezierControlPointsCellKernel(CudaMat2<float> controlPointZs, CudaMat2<float> bezierControlPointZs);

void cudaComputeBezierControlPointsOnDevice(CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs);
bool cudaComputeBezierControlPoints(int deviceId, CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs);
bool cudaEvalBSpline(int deviceId, CudaMat2<float>& controlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat);
