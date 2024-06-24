#include <cstdio>
#include <stdexcept>
#include <exception>
#include <stdio.h>
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cudaUtil.h"
#include "cudaBSplineEval.h"
#include "cudaBSplineCoeffs.h"
#include "cudaBSplineTransform.h"


// copied from cv-remap-poly-cuda
// -------------------------------------

/**
* @brief Single cubic interpolation at location t between four values.
*/
__device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t)
{
    float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float a2 = -0.5f * p0 + 0.5f * p2;
    float a3 = p1;

    float t2 = t * t;
    return a0 * t2 * t + a1 * t2 + a2 * t + a3;
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

/**
* @brief Sample a single point from the specified source texture using bicubic interpolation.
*/
__device__ float bicubicSample(cudaTextureObject_t srcTexture, int width, int height, float x, float y)
{
    int x1 = clamp(static_cast<int>(floor(x)) - 1, 0, width - 1);
    int y1 = clamp(static_cast<int>(floor(y)) - 1, 0, height - 1);
    int x2 = clamp(x1 + 1, 0, width - 1);
    int y2 = clamp(y1 + 1, 0, height - 1);
    int x3 = clamp(x1 + 2, 0, width - 1);
    int y3 = clamp(y1 + 2, 0, height - 1);
    int x4 = clamp(x1 + 3, 0, width - 1);
    int y4 = clamp(y1 + 3, 0, height - 1);

    float dx = x - x2;
    float dy = y - y2;

    float row_interpolations[4];

    for (int i = 0; i < 4; ++i)
    {
        float p1 = (float)tex2D<uint8_t>(srcTexture, x1, y1 + i);
        float p2 = (float)tex2D<uint8_t>(srcTexture, x2, y1 + i);
        float p3 = (float)tex2D<uint8_t>(srcTexture, x3, y1 + i);
        float p4 = (float)tex2D<uint8_t>(srcTexture, x4, y1 + i);

        row_interpolations[i] = cubicInterpolate(p1, p2, p3, p4, dx);
    }

    return cubicInterpolate(row_interpolations[0], row_interpolations[1], row_interpolations[2], row_interpolations[3], dy);
}

// -------------------------------------

__device__ int clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ float4 cubicInterpolateFloat4(uchar4 p0, uchar4 p1, uchar4 p2, uchar4 p3, float t)
{
    float4 val;
    val.x = cubicInterpolate(p0.x, p1.x, p2.x, p3.x, t);
    val.y = cubicInterpolate(p0.y, p1.y, p2.y, p3.y, t);
    val.z = cubicInterpolate(p0.z, p1.z, p2.z, p3.z, t);
    val.w = cubicInterpolate(p0.w, p1.w, p2.w, p3.w, t);
    return val;
}

__device__ float4 cubicInterpolateFloat4(float4 p0, float4 p1, float4 p2, float4 p3, float t)
{
    float4 val;
    val.x = cubicInterpolate(p0.x, p1.x, p2.x, p3.x, t);
    val.y = cubicInterpolate(p0.y, p1.y, p2.y, p3.y, t);
    val.z = cubicInterpolate(p0.z, p1.z, p2.z, p3.z, t);
    val.w = cubicInterpolate(p0.w, p1.w, p2.w, p3.w, t);
    return val;
}

/**
* @brief Sample a single point from the specified source texture using bicubic interpolation.
*/
__device__ float4 bicubicSampleFloat4(cudaTextureObject_t srcTexture, int width, int height, float x, float y)
{
    int x1 = clamp(static_cast<int>(floor(x)) - 1, 0, width - 1);
    int y1 = clamp(static_cast<int>(floor(y)) - 1, 0, height - 1);
    int x2 = clamp(x1 + 1, 0, width - 1);
    int y2 = clamp(y1 + 1, 0, height - 1);
    int x3 = clamp(x1 + 2, 0, width - 1);
    int y3 = clamp(y1 + 2, 0, height - 1);
    int x4 = clamp(x1 + 3, 0, width - 1);
    int y4 = clamp(y1 + 3, 0, height - 1);

    float dx = x - x2;
    float dy = y - y2;

    float4 row_interpolations[4];

    for (int i = 0; i < 4; ++i)
    {
        // nearest-neighbor sampling
        uchar4 p1 = tex2D<uchar4>(srcTexture, x1, y1 + i);
        uchar4 p2 = tex2D<uchar4>(srcTexture, x2, y1 + i);
        uchar4 p3 = tex2D<uchar4>(srcTexture, x3, y1 + i);
        uchar4 p4 = tex2D<uchar4>(srcTexture, x4, y1 + i);

        row_interpolations[i] = cubicInterpolateFloat4(p1, p2, p3, p4, dx);
    }

    return cubicInterpolateFloat4(row_interpolations[0], row_interpolations[1], row_interpolations[2], row_interpolations[3], dy);
}


__global__ void transformKernel(CudaMat2<uint8_t> outputImage, int samplingType, 
    CudaMat2<float> dxBezierControlPointZs, float dxScale, 
    CudaMat2<float> dyBezierControlPointZs, float dyScale, cudaTextureObject_t srcTexture)
{
    // which output pixel this thread is for
    unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if ((yi >= outputImage.rows) || (xi >= outputImage.cols))
    {
        return;
    }

    // Eval bezier surface at a point for this pixel.
    float cellx = (float)xi / dxScale + 1.0f; // +1 because first cells are not computable
    float celly = (float)yi / dyScale + 1.0f;
    CudaPoint3<float> dxPt = evalBezierSurfaceCubicPointSub(dxBezierControlPointZs, cellx, celly);
    CudaPoint3<float> dyPt = evalBezierSurfaceCubicPointSub(dyBezierControlPointZs, cellx, celly);

    // The bezier values are offsets in pixels for where to sample from the source image.
    float x = xi + dxPt.z;
    float y = yi + dyPt.z;

    // sample from the computed location
    if (samplingType == 0)
    {
        // nearest neighbor
        outputImage.dataDevice[yi * outputImage.stride + xi] = tex2D<uint8_t>(srcTexture, x + 0.5f, y + 0.5f);
    }
    else if (samplingType == 1)
    {
        // bilinear, returns float
        float val = tex2D<float>(srcTexture, x + 0.5f, y + 0.5f);
        outputImage.dataDevice[yi * outputImage.stride + xi] = (uint8_t)(val * 255.0f);
    }
    else if (samplingType == 2)
    {
        // bicubic
        // TEMP: outputImage.dataDevice[yi * outputImage.stride + xi] = (uint8_t)123;

        // need to clamp to border to avoid artifacts
        const int borderPx = 2;
        int width = outputImage.cols;
        int height = outputImage.rows;

        if ((x >= borderPx) && (x < width - borderPx) && (y >= borderPx) && (y < height - borderPx))
        {
            float b = bicubicSample(srcTexture, width, height, x, y);
            int bi = (int)(b + 0.5f);
            bi = max((int)0, min((int)255, bi)); // clamp
            outputImage.dataDevice[yi * outputImage.stride + xi] = (uint8_t)bi;
        }
        else
        {
            // just take the nearest border pixel
            uint8_t val = tex2D<uint8_t>(srcTexture, x + 0.5f, y + 0.5f);
            outputImage.dataDevice[yi * outputImage.stride + xi] = val;
        }
    }
}

__global__ void transformKernelBgra(CudaMat2<BgraQuad> outputImage, int samplingType, 
    CudaMat2<float> dxBezierControlPointZs, float dxScale, 
    CudaMat2<float> dyBezierControlPointZs, float dyScale, cudaTextureObject_t srcTexture)
{
    // which output pixel this thread is for
    unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if ((yi >= outputImage.rows) || (xi >= outputImage.cols))
    {
        return;
    }

    // Eval bezier surface at a point for this pixel.
    float cellx = (float)xi / dxScale + 1.0f; // +1 because first cells are not computable
    float celly = (float)yi / dyScale + 1.0f;
    CudaPoint3<float> dxPt = evalBezierSurfaceCubicPointSub(dxBezierControlPointZs, cellx, celly);
    CudaPoint3<float> dyPt = evalBezierSurfaceCubicPointSub(dyBezierControlPointZs, cellx, celly);

    // The bezier values are offsets in pixels for where to sample from the source image.
    float x = xi - dxPt.z;
    float y = yi - dyPt.z;

    // sample from the computed location
    if (samplingType == 0)
    {
        // nearest neighbor
        x += 0.5f; // sampling offset
        y += 0.5f;

        uchar4 val = tex2D<uchar4>(srcTexture, x, y);
        outputImage.dataDevice[yi * outputImage.stride + xi].b = val.x;
        outputImage.dataDevice[yi * outputImage.stride + xi].g = val.y;
        outputImage.dataDevice[yi * outputImage.stride + xi].r = val.z;
        outputImage.dataDevice[yi * outputImage.stride + xi].a = val.w;
    }
    else if (samplingType == 1)
    {
        // bilinear
        x += 0.5f; // sampling offset
        y += 0.5f;

        // bilinear, returns float
        float4 val = tex2D<float4>(srcTexture, x, y);

        outputImage.dataDevice[yi * outputImage.stride + xi].b = (uint8_t)(val.x * 255.0f);
        outputImage.dataDevice[yi * outputImage.stride + xi].g = (uint8_t)(val.y * 255.0f);
        outputImage.dataDevice[yi * outputImage.stride + xi].r = (uint8_t)(val.z * 255.0f);
        outputImage.dataDevice[yi * outputImage.stride + xi].a = (uint8_t)(val.w * 255.0f);
    }
    else if (samplingType == 2)
    {
        // bicubic

        // need to clamp to border to avoid artifacts
        const int borderPx = 2;
        int width = outputImage.cols;
        int height = outputImage.rows;

        if ((x >= borderPx) && (x < width - borderPx) && (y >= borderPx) && (y < height - borderPx))
        {
            float4 val = bicubicSampleFloat4(srcTexture, width, height, x, y);
            outputImage.dataDevice[yi * outputImage.stride + xi].b = (uint8_t)clamp(val.x + 0.5f, 0.0f, 255.9f);
            outputImage.dataDevice[yi * outputImage.stride + xi].g = (uint8_t)clamp(val.y + 0.5f, 0.0f, 255.9f);
            outputImage.dataDevice[yi * outputImage.stride + xi].r = (uint8_t)clamp(val.z + 0.5f, 0.0f, 255.9f);
            outputImage.dataDevice[yi * outputImage.stride + xi].a = (uint8_t)clamp(val.w + 0.5f, 0.0f, 255.9f);
        }
        else
        {
            // just take the nearest border pixel
            x += 0.5f; // sampling offset
            y += 0.5f;

            uchar4 val = tex2D<uchar4>(srcTexture, x, y);
            outputImage.dataDevice[yi * outputImage.stride + xi].b = val.x;
            outputImage.dataDevice[yi * outputImage.stride + xi].g = val.y;
            outputImage.dataDevice[yi * outputImage.stride + xi].r = val.z;
            outputImage.dataDevice[yi * outputImage.stride + xi].a = val.w;
        }
    }
}

/**
 * @brief Transform an image using a B-spline transformation.
 * This doesn't clean up on error.
 * @param inputImage 
 * @param dxControlPointZs B-Spline control points for the x-axis.
 * @param dxScale Horizontal scale of the B-Spline transformation vs image size, in units of pixels per cell.
 * @param dyControlPointZs B-Spline control points for the y-axis.
 * @param dyScale Vertical scale of the B-Spline transformation vs image size, in units of pixels per cell.
 * @param samplingType 0=NN, 1=bilinear, 2=bicubic
 * @param outputImage 
 * @return 
 */
cudaError_t cudaBSplineTransformImage(int deviceId, CudaMat2<uint8_t>& inputImage, CudaMat2<float>& dxControlPointZs, 
    float dxScale, CudaMat2<float>& dyControlPointZs, float dyScale, int samplingType, 
    CudaMat2<uint8_t>& outputImage)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    // For eval, one thread per output pixel.
    dim3 dimBlockEval, dimGridEval;
    computeGridAndBlockDims(outputImage.rows, outputImage.cols, dimGridEval, dimBlockEval);

    // src image
    cudaArray_t srcArray = {};
    cudaTextureObject_t srcTexture = {};
    setupSrcImageTexture8u(inputImage, samplingType, srcArray, srcTexture);

    // dst image
    cudaStatus = outputImage.cudaMalloc();
    assertCudaStatus(cudaStatus, "CUDA malloc output image failed");

    // coeffs
    CudaMat2<float> dxBezierControlPointZs, dyBezierControlPointZs;
    cudaComputeBezierControlPointsOnDevice(dxControlPointZs, dxBezierControlPointZs); // no synchronize, no copy to host
    cudaComputeBezierControlPointsOnDevice(dyControlPointZs, dyBezierControlPointZs); // no synchronize, no copy to host

    // have to wait, we need those bezier points before can eval with them
    cudaStatus = cudaDeviceSynchronize();
    assertCudaStatus(cudaStatus, "cudaDeviceSynchronize failed");

    // kernel
    // clang-format off
    transformKernel<<<dimGridEval, dimBlockEval, 0>>>(outputImage, samplingType, dxBezierControlPointZs, dxScale, dyBezierControlPointZs, dyScale, srcTexture);
    // clang-format on

    cudaStatus = cudaGetLastError();
    assertCudaStatus(cudaStatus, "kernel launch failed");

    // wait for the kernel to finish and check errors
    cudaStatus = cudaDeviceSynchronize();
    assertCudaStatus(cudaStatus, "cudaDeviceSynchronize failed");

    // output
    cudaStatus = outputImage.copyDeviceToHost();
    assertCudaStatus(cudaStatus, "cudaMemcpy dest image failed");

    cudaDestroyTextureObject(srcTexture);
    cudaFreeArray(srcArray);

    inputImage.cudaFree();
    outputImage.cudaFree();

    return cudaSuccess;
}

cudaError_t cudaBSplineTransformImage(int deviceId, CudaMat2<BgraQuad>& inputImage, CudaMat2<float>& dxControlPointZs, 
    float dxScale, CudaMat2<float>& dyControlPointZs, float dyScale, int samplingType, 
    CudaMat2<BgraQuad>& outputImage)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    // For eval, one thread per output pixel.
    dim3 dimBlockEval, dimGridEval;
    computeGridAndBlockDims(outputImage.rows, outputImage.cols, dimGridEval, dimBlockEval);

    // src image
    cudaArray_t srcArray = {};
    cudaTextureObject_t srcTexture = {};
    setupSrcImageTextureBgra(inputImage, samplingType, srcArray, srcTexture);

    // dst image
    cudaStatus = outputImage.cudaMalloc();
    assertCudaStatus(cudaStatus, "CUDA malloc output image failed");

    // coeffs
    CudaMat2<float> dxBezierControlPointZs, dyBezierControlPointZs;
    cudaComputeBezierControlPointsOnDevice(dxControlPointZs, dxBezierControlPointZs); // no synchronize, no copy to host
    cudaComputeBezierControlPointsOnDevice(dyControlPointZs, dyBezierControlPointZs); // no synchronize, no copy to host

    // have to wait, we need those bezier points before can eval with them
    cudaStatus = cudaDeviceSynchronize();
    assertCudaStatus(cudaStatus, "cudaDeviceSynchronize failed");

    // kernel
    // clang-format off
    transformKernelBgra<<<dimGridEval, dimBlockEval, 0>>>(outputImage, samplingType, dxBezierControlPointZs, dxScale, dyBezierControlPointZs, dyScale, srcTexture);
    // clang-format on

    cudaStatus = cudaGetLastError();
    assertCudaStatus(cudaStatus, "kernel launch failed");

    // wait for the kernel to finish and check errors
    cudaStatus = cudaDeviceSynchronize();
    assertCudaStatus(cudaStatus, "cudaDeviceSynchronize failed");

    // output
    cudaStatus = outputImage.copyDeviceToHost();
    assertCudaStatus(cudaStatus, "cudaMemcpy dest image failed");

    cudaDestroyTextureObject(srcTexture);
    cudaFreeArray(srcArray);

    inputImage.cudaFree();
    outputImage.cudaFree();

    return cudaSuccess;
}
