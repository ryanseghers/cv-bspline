#include <cstdio>
#include <exception>
#include <stdio.h>
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cudaUtil.h"
#include "cudaBSplineEval.h"
#include "cudaBSplineCoeffs.h"


__device__ float bezierPolyTerm(int i, float t) 
{
    float u = 1 - t;

    switch (i) 
    {
        case 0: return u*u*u;
        case 1: return 3*t*u*u;
        case 2: return 3*t*t*u;
        case 3: return t*t*t;
    }

    return 0;
}

/**
 * @brief 
 * @param pzs Pointer to first control point Z value.
 * @param zStride Stride between control points in Z.
 * @param u 
 * @param v 
 * @return 
 */
__device__ CudaPoint3<float> evalBezierSurfaceCubicPoint(float* pzs, int zStride, float u, float v) 
{
    const float THIRDS[] = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    //float u1 = 1.0f - u;
    float v1 = 1.0f - v;
    float v13 = v1 * v1 * v1;
    float v12 = 3 * v * v1 * v1;
    float v21 = 3 * v * v * v1;
    float v3 = v * v * v;

    for (int i = 0; i < 4; ++i) 
    {
        float* pz = &pzs[i * zStride];
        float tx;
        float ty = THIRDS[i];
        float biu = bezierPolyTerm(i, u);

        float bjv = v13;
        //tx = 0.0f;
        float tz = pz[0];
        //x += biu * bjv * tx;
        y += biu * bjv * ty;
        z += biu * bjv * tz;

        bjv = v12;
        tx = THIRDS[1];
        tz = pz[1];
        x += biu * bjv * tx;
        y += biu * bjv * ty;
        z += biu * bjv * tz;

        bjv = v21;
        tx = THIRDS[2];
        tz = pz[2];
        x += biu * bjv * tx;
        y += biu * bjv * ty;
        z += biu * bjv * tz;

        bjv = v3;
        //tx = 1.0f;
        tz = pz[3];
        x += biu * bjv; // * tx
        y += biu * bjv * ty;
        z += biu * bjv * tz;
    }

    CudaPoint3<float> pt;
    pt.x = x;
    pt.y = y;
    pt.z = z;
    return pt;
}

/**
 * @brief Eval bezier surface at a float grid point. This finds the cell and then evaluates the surface at the point within the cell.
 * @param bezierControlPointZs 3x3 per cell
 * @param bzGridX In grid coords.
 * @param bzGridY In grid coords.
 * @return 
 */
__device__ CudaPoint3<float> evalBezierSurfaceCubicPointSub(CudaMat2<float> bezierControlPointZs, float bzGridX, float bzGridY)
{
    int cellx = (int)bzGridX;
    int celly = (int)bzGridY;

    // check in bounds
    int gridRows = bezierControlPointZs.rows / 3;
    int gridCols = bezierControlPointZs.cols / 3;

    if ((cellx < 0) || (cellx >= gridCols - 1) || (celly < 0) || (celly >= gridRows - 1))
    {
        CudaPoint3<float> pt;
        pt.x = 0.0f;
        pt.y = 0.0f;
        pt.z = 0.0f;
        return pt;
    }

    // indices into zs for this cell
    // 3 control points per cell
    int zsxi = celly * bezierControlPointZs.stride * 3 + cellx * 3; // this is the index of the first coeff for this cell

    // u, v fractions in cell
    float v = bzGridX - cellx;
    float u = bzGridY - celly;

    // evaluate the bezier surface
    float* zs = bezierControlPointZs.dataDevice;
    CudaPoint3<float> pt = evalBezierSurfaceCubicPoint(&zs[zsxi], bezierControlPointZs.stride, u, v);

    // back to image coords
    pt.x += cellx;
    pt.y += celly;

    return pt;
}

/**
* @brief Kernel to evaluate the b-spline surface at the specified output pixel, from precomputed bezier control points.
* @param dstMat Output image, contiguous mem, 3 floats per pixel.
* @param gridWidth B-spline grid width, in cells.
* @param gridHeight B-spline grid height, in cells.
* @param zs Control point Z values, contiguous mem.
*/
__global__ void evalBSplineKernelPrecomp(CudaMat2<CudaPoint3<float>> dstMat, int gridWidth, int gridHeight, CudaMat2<float> bezierControlPointZs, int nPointsPerDim)
{
    // which output pixel this thread is for
    unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if ((yi >= dstMat.rows) || (xi >= dstMat.cols))
    {
        return;
    }

    // which grid cell this pixel is in
    float x = (float)xi / nPointsPerDim;
    float y = (float)yi / nPointsPerDim;

    CudaPoint3<float> pt = evalBezierSurfaceCubicPointSub(bezierControlPointZs, x, y);

    // output
    int idx = yi * dstMat.cols + xi;
    dstMat.dataDevice[idx] = pt;

    //printf("evalBSplineKernel: xi: %d, yi: %d, idx: %d\n", xi, yi, idx);
}

/**
 * @brief Eval the b-spline surface using precomputed bezier control points.
 */
bool cudaEvalBSplinePrecomp(int deviceId, CudaMat2<float>& bezierControlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    // 3 points per cell, plus 1 for the border, and often 3 for the border with the last 2 unused
    int nCellRows = (bezierControlPointZs.rows - 1) / 3;
    int nCellCols = (bezierControlPointZs.cols - 1) / 3;

    int outputHeight = nCellRows * nPointsPerDim;
    int outputWidth = nCellCols * nPointsPerDim;

    // threads per block (16 * 8 == 128 which is number of cuda cores per SM?)
    // TEMP: 1 thread per block for small tests
    dim3 dimBlock(1, 1, 1);

    // however many thread blocks to cover image
    // (if image is not multiple of thread block dims then some will not be computed)
    dim3 dimGrid(outputWidth / dimBlock.x, outputHeight / dimBlock.y, 1); 

    // dst image
    cudaStatus = outputMat.cudaMalloc();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dest image failed!");
        goto Error;
    }

    cudaStatus = bezierControlPointZs.copyHostToDevice(); // also alloc

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc and copy bSplineControlPointZs to device failed!");
        goto Error;
    }

    // kernel
    // clang-format off
    evalBSplineKernelPrecomp<<<dimGrid, dimBlock, 0>>>(outputMat, nCellCols, nCellRows, bezierControlPointZs, nPointsPerDim);
    // clang-format on

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // wait for the kernel to finish and check errors
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Get results
    cudaStatus = outputMat.copyDeviceToHost();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dest image failed!");
        goto Error;
    }

Error:
    outputMat.cudaFree();
    bezierControlPointZs.cudaFree();

    return cudaStatus == cudaSuccess;
}

/**
* @brief Evaluate the uniform b-spline surface defined by the control points, with nPointsPerDim * nPointsPerDim per cell.
* @param bSplineControlPointZs B-spline control point Z values.
* @param nPointsPerDim How finely to evaluate the surface, in points per dimension per cell.
* @param outputMat Pre-allocated output matrix with (nPointsPerDim * cells) x (nPointsPerDim * cells) matrix of computed values.
* @return Result code.
*/
bool cudaEvalBSpline(int deviceId, CudaMat2<float>& bSplineControlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    int nCellRows = bSplineControlPointZs.rows;
    int nCellCols = bSplineControlPointZs.cols;

    int outputHeight = nCellRows * nPointsPerDim;
    int outputWidth = nCellCols * nPointsPerDim;

    // One thread per cell (per input coeff, not per output coeff).
    dim3 dimBlockCoeffs, dimGridCoeffs;
    computeGridAndBlockDims(nCellRows, nCellCols, dimGridCoeffs, dimBlockCoeffs);

    // For eval, one thread per output pixel.
    dim3 dimBlockEval, dimGridEval;
    computeGridAndBlockDims(outputHeight, outputWidth, dimGridEval, dimBlockEval);

    //
    // Compute bezier control points
    //
    CudaMat2<float> bezierControlPointZs(nCellRows * 3, nCellCols * 3);
    cudaStatus = bezierControlPointZs.cudaMalloc();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc bezierControlPointZs failed!");
        goto Error;
    }

    cudaStatus = bSplineControlPointZs.copyHostToDevice(); // also does alloc

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc and copy bSplineControlPointZs to device failed!");
        goto Error;
    }

    // clang-format off
    computeBezierControlPointsCellKernel<<<dimGridCoeffs, dimBlockCoeffs, 0>>>(bSplineControlPointZs, bezierControlPointZs);
    // clang-format on


    //
    // Eval
    //

    // dst image
    cudaStatus = outputMat.cudaMalloc();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dest image failed!");
        goto Error;
    }

    // kernel
    // clang-format off
    evalBSplineKernelPrecomp<<<dimGridEval, dimBlockEval, 0>>>(outputMat, nCellCols, nCellRows, bezierControlPointZs, nPointsPerDim);
    // clang-format on

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // wait for the kernel to finish and check errors
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Get results
    cudaStatus = outputMat.copyDeviceToHost();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dest image failed!");
        goto Error;
    }

Error:
    outputMat.cudaFree();
    bezierControlPointZs.cudaFree();

    return cudaStatus == cudaSuccess;
}
