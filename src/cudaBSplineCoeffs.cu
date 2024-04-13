#include <cstdio>
#include <exception>
#include <stdio.h>
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cudaUtil.h"
#include "cudaBSplineCoeffs.h"


// shared with BSplineMat.cpp
__device__ void horizontalBSplineInterpFull(float* bsp, int bspStride, int r, int c, int numCols, float* pOut)
{
    const float oneThird = 1.0f / 3.0f;
    const float twoThird = 2.0f / 3.0f;

    float zThis = bsp[r * bspStride + c];

    if (c == 0)
    {
        // first cell col
        pOut[0] = zThis;

        float zNext = bsp[r * bspStride + c + 1];
        pOut[1] = interpolate(zThis, zNext, oneThird);
        pOut[2] = interpolate(zThis, zNext, twoThird);
    }
    else if (c == numCols - 1)
    {
        // last cell col
        pOut[0] = zThis;
        pOut[1] = 0.0f;
        pOut[2] = 0.0f;
    }
    else
    {
        // middle cell col
        float zPrior = bsp[r * bspStride + c - 1];
        float zNext = bsp[r * bspStride + c + 1];
        float p1 = interpolate(zPrior, zThis, twoThird);
        float p2 = interpolate(zThis, zNext, oneThird);
        float p3 = interpolate(zThis, zNext, twoThird);

        float mid = interpolate(p1, p2, 0.5f);
        pOut[0] = mid;
        pOut[1] = p2;
        pOut[2] = p3;
    }
}

/**
* @brief Both directions, just one cell, not referencing other cells' results.
* This can be called on edge cells.
*/
__device__ void computeBezierControlPointsSingleCellIsolatedFull(CudaMat2<float> bSplineControlPointZs, int r, int c, CudaMat2<float> bezierControlPointsZs)
{
    // These are fractions of a cell, and right now we are using 1-dim cells.
    const float oneThird = 1.0f / 3.0f;
    const float twoThird = 2.0f / 3.0f;

    // Setup for indexing the matrices.
    int bri = r * 3; // bezier row index
    int bci = c * 3; // bezier col index

    int rMax = bSplineControlPointZs.rows - 1;

    float* bsp = (float*)bSplineControlPointZs.dataDevice;
    int bspNumCols = bSplineControlPointZs.cols;
    int bspStride = bSplineControlPointZs.stride;

    float* bzp = (float*)bezierControlPointsZs.dataDevice;
    int bzpStride = bezierControlPointsZs.stride;

    // Do horizontal into 3 tmp rows.
    // (only using 3 cols but pad for size)
    float bzPriorRow[4] = { 0.0f };
    float bzThisRow[4] = { 0.0f };
    float bzNextRow[4] = { 0.0f };

    if (r > 0)
    {
        horizontalBSplineInterpFull(bsp, bspStride, r - 1, c, bspNumCols, &bzPriorRow[0]);
    }

    horizontalBSplineInterpFull(bsp, bspStride, r, c, bspNumCols, &bzThisRow[0]);

    if (r < rMax)
    {
        horizontalBSplineInterpFull(bsp, bspStride, r + 1, c, bspNumCols, &bzNextRow[0]);
    }

    // don't do last two cols of last cell column
    int biMax = (c == bSplineControlPointZs.cols - 1) ? 1 : 3;

    // Vertical
    for (int bi = 0; bi < biMax; bi++) // bezier col
    {
        float bptThis = bzThisRow[bi];

        if (r == 0)
        {
            // first row
            bzp[bzpStride * (bri + 0) + bci + bi] = bptThis;

            float bptNext = bzNextRow[bi];
            float p2 = interpolate(bptThis, bptNext, oneThird);
            bzp[bzpStride * (bri + 1) + bci + bi] = p2;

            float p3 = interpolate(bptThis, bptNext, twoThird);
            bzp[bzpStride * (bri + 2) + bci + bi] = p3;
        }
        else if (r == rMax)
        {
            // last row
            bzp[bzpStride * (bri + 0) + bci + bi] = bptThis;
        }
        else
        {
            // middle rows
            float bptPrior = bzPriorRow[bi];
            float bptNext = bzNextRow[bi];

            float p1 = interpolate(bptPrior, bptThis, twoThird);
            float p2 = interpolate(bptThis, bptNext, oneThird);
            float p3 = interpolate(bptThis, bptNext, twoThird);

            float mid = interpolate(p1, p2, 0.5f);
            bzp[bzpStride * (bri + 0) + bci + bi] = mid;
            bzp[bzpStride * (bri + 1) + bci + bi] = p2;
            bzp[bzpStride * (bri + 2) + bci + bi] = p3;
        }
    }
}

/**
* @brief Compute Bezier control points from the B-spline control points for a single cell.
* There are 3x3 bezier control points per cell.
*/
__global__ void computeBezierControlPointsCellKernel(CudaMat2<float> controlPointZs, CudaMat2<float> bezierControlPointZs)
{
    // which cell this thread is for
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if ((r * 3 >= bezierControlPointZs.rows) || (c * 3 >= bezierControlPointZs.cols))
    {
        return;
    }

    computeBezierControlPointsSingleCellIsolatedFull(controlPointZs, r, c, bezierControlPointZs);
}

/**
 * @brief Compute, don't synchronize, and leave coeffs on-device.
 */
void cudaComputeBezierControlPointsOnDevice(CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs)
{
    cudaError_t cudaStatus = cudaSuccess;

    // Number of cells.
    // (We include top/left edge points/cells just for simplicity in coding, but they are not computable and not used.)
    int cellRows = controlPointZs.rows;
    int cellCols = controlPointZs.cols;

    dim3 dimBlock, dimGrid;
    computeGridAndBlockDims(cellRows, cellCols, dimBlock, dimGrid);

    // this doesn't resize, just initial set size
    bezierControlPointZs.setSize(cellRows * 3, cellCols * 3);

    // dst image
    cudaStatus = bezierControlPointZs.cudaMalloc();
    assertCudaStatus(cudaStatus, "cudaMalloc dest image failed");

    cudaStatus = controlPointZs.copyHostToDevice(); // also does alloc
    assertCudaStatus(cudaStatus, "cudaMalloc and copy controlPointZs to device failed");

    // clang-format off
    computeBezierControlPointsCellKernel<<<dimGrid, dimBlock, 0>>>(controlPointZs, bezierControlPointZs);
    // clang-format on

    cudaStatus = cudaGetLastError();
    assertCudaStatus(cudaStatus, "kernel launch failed");
}

/**
* @brief Compute bezier control points from the b-spline control points.
* This is the version that does horizontal and vertical in separate kernels.
* There are 3x3 bezier control points per cell.
* Use one thread per cell.
* This doesn't clean up on error.
* @param deviceId 
* @param controlPointZs B-spline control points.
* @param bezierControlPointZs float, 3x3 times the size of controlPointZs.
* @return Always true because this throws on error.
*/
bool cudaComputeBezierControlPointsCell(int deviceId, CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs)
{
    cudaError_t cudaStatus = cudaSuccess;

    // setting device multiple times has no impact
    setDevice(deviceId);

    cudaComputeBezierControlPointsOnDevice(controlPointZs, bezierControlPointZs);

    // wait for the kernel to finish and check errors
    cudaStatus = cudaDeviceSynchronize();
    assertCudaStatus(cudaStatus, "cudaDeviceSynchronize");

    // Get results
    cudaStatus = bezierControlPointZs.copyDeviceToHost();
    assertCudaStatus(cudaStatus, "cudaMemcpy dest image");

    bezierControlPointZs.cudaFree();
    controlPointZs.cudaFree();

    return cudaStatus == cudaSuccess;
}

/**
* @brief Compute bezier control points from the b-spline control points.
* There are 3x3 bezier control points per cell.
* Use one thread per cell.
* @param deviceId 
* @param controlPointZs B-spline control points.
* @param bezierControlPointZs float, 3x3 times the size of controlPointZs.
* @return Success or not.
*/
bool cudaComputeBezierControlPoints(int deviceId, CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs)
{
    return cudaComputeBezierControlPointsCell(deviceId, controlPointZs, bezierControlPointZs);
}
