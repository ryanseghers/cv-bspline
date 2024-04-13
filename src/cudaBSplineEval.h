#pragma once
#include "cuda_runtime.h"
#include "cudaUtil.h"


__device__ CudaPoint3<float> evalBezierSurfaceCubicPointSub(CudaMat2<float> bezierControlPointZs, float bzGridX, float bzGridY);
bool cudaEvalBSplinePrecomp(int deviceId, CudaMat2<float>& bezierControlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat);
bool cudaEvalBSpline(int deviceId, CudaMat2<float>& bSplineControlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat);
