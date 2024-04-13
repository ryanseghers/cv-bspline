#pragma once
#include "cuda_runtime.h"
#include "cudaUtil.h"

/**
 * @brief Apply b-spline transformation to an image.
 * @param deviceId 
 * @param inputImage 
 * @param dxControlPointZs 
 * @param dxScale Pixels per cell. This specifies where in the control point grid to evaluate the bezier surface to determine dx.
 * @param dyControlPointZs 
 * @param dyScale Pixels per cell. This specifies where in the control point grid to evaluate the bezier surface to determine dx.
 * @param samplingType 
 * @param outputImage 
 * @return 
 */
cudaError_t cudaBSplineTransformImage(int deviceId, CudaMat2<uint8_t>& inputImage, CudaMat2<float>& dxControlPointZs, float dxScale, 
    CudaMat2<float>& dyControlPointZs, float dyScale, int samplingType, CudaMat2<uint8_t>& outputImage);

/**
 * @brief Copy of the above function, but for RGB images.
 */
cudaError_t cudaBSplineTransformImage(int deviceId, CudaMat2<BgraQuad>& inputImage, CudaMat2<float>& dxControlPointZs,
    float dxScale, CudaMat2<float>& dyControlPointZs, float dyScale, int samplingType,
    CudaMat2<BgraQuad>& outputImage);
