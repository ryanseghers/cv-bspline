#include <cstdio>
#include <exception>
#include <stdio.h>
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cudaUtil.h"
#include "cudaBezier.h"


/**
* @brief Eval the cubic polynomial.
*/
__device__ float evalPoly(int x, int y, float* coeffs)
{
    // 1, x, y, x^2, xy, y^2, x^3, x^2*y, y^2*x, y^3
    float x2 = x * x;
    float y2 = y * y;

    return coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * y
        + coeffs[3] * x2
        + coeffs[4] * x * y
        + coeffs[5] * y2
        + coeffs[6] * x2 * x
        + coeffs[7] * x2 * y
        + coeffs[8] * y2 * x
        + coeffs[9] * y2 * y;
}

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
        float p1 = (float)tex2D<uint16_t>(srcTexture, x1, y1 + i);
        float p2 = (float)tex2D<uint16_t>(srcTexture, x2, y1 + i);
        float p3 = (float)tex2D<uint16_t>(srcTexture, x3, y1 + i);
        float p4 = (float)tex2D<uint16_t>(srcTexture, x4, y1 + i);

        row_interpolations[i] = cubicInterpolate(p1, p2, p3, p4, dx);
    }

    return cubicInterpolate(row_interpolations[0], row_interpolations[1], row_interpolations[2], row_interpolations[3], dy);
}

/**
* @brief Horizontal, just one cell.
* This can be called for first and last rows, but not for first and last cols.
*/
__device__ void computeBezierControlPointsSingleCellHorizontal(const CudaMat2<float> bSplineControlPointZs, int r, int c, CudaMat2<float> bezierControlPointsZs)
{
    // These are fractions of a cell, and right now we are using 1-dim cells.
    const float oneThird = 1.0f / 3.0f;
    const float twoThird = 2.0f / 3.0f;

    // Setup for indexing the matrices.
    float* bsp = (float*)bSplineControlPointZs.dataDevice;
    int bspStride = bSplineControlPointZs.stride;

    float* bzp = (float*)bezierControlPointsZs.dataDevice;
    int bzpStride = bezierControlPointsZs.stride;

    // Compute
    float zPrior = bsp[r * bspStride + c - 1];
    float zThis = bsp[r * bspStride + c];
    float zNext = bsp[r * bspStride + c + 1]; // By our definition of a cell there is always a next point.

    float p2z = interpolate(zThis, zNext, oneThird);
    float p3z = interpolate(zThis, zNext, twoThird);
    float p1z = interpolate(zPrior, zThis, twoThird);
    float midz = interpolate(p1z, p2z, 0.5f);
    bzp[r * 3 * bzpStride + c * 3] = midz;
    bzp[r * 3 * bzpStride + c * 3 + 1] = p2z;
    bzp[r * 3 * bzpStride + c * 3 + 2] = p3z;
}

/**
* THIS DOESN'T WORK, CANNOT LOOK AT OTHER CELLS.
* @brief Vertical, just one cell.
*/
__device__ void computeBezierControlPointsSingleCellVertical(int r, int c, CudaMat2<float> bezierControlPointsZs)
{
    // These are fractions of a cell, and right now we are using 1-dim cells.
    const float oneThird = 1.0f / 3.0f;
    const float twoThird = 2.0f / 3.0f;

    // Setup for indexing the matrices.
    float* bzp = (float*)bezierControlPointsZs.dataDevice;
    int bzpStride = bezierControlPointsZs.stride;

    // Compute
    int bri = r * 3; // bezier row index

    // Cells have 3 control points so have to interpolate each of them.
    for (int bi = 0; bi < 3; bi++)
    {
        float x = c + bi * oneThird; // uniform grid
        int bci = c * 3 + bi; // bezier col index

        float bptThis = bzp[bzpStride * bri + bci];

        // By our definition of a cell there is always a next point.
        float bptNext = bzp[bzpStride * (bri + 3) + bci];

        // By definition of a cell there is always a next point.
        float p2 = interpolate(bptThis, bptNext, oneThird);
        float p3 = interpolate(bptThis, bptNext, twoThird);

        // First bezier point in cell is the mid-point from prior two-thirds point to this one-third point.
        if (r > 0)
        {
            float bptPrior = bzp[bzpStride * (bri - 3) + bci];
            float p1 = interpolate(bptPrior, bptThis, twoThird);
            float mid = interpolate(p1, p2, 0.5f);
            bzp[bzpStride * bri + bci] = mid;
        }

        bzp[bzpStride * (bri + 1) + bci] = p2;
        bzp[bzpStride * (bri + 2) + bci] = p3;
    }
}

/**
 * @brief Compute Bezier control points from the B-spline control points for a single cell.
 * This does either horitontal or vertical, because all horizontal needs to be done before any vertical.
 * There are 3x3 bezier control points per cell.
 */
__global__ void computeBezierControlPointsKernelSeparated(CudaMat2<float> controlPointZs, CudaMat2<float> bezierControlPointZs, bool doHorizontal)
{
    // which cell this thread is for
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if ((r * 3 >= bezierControlPointZs.rows) || (c * 3 >= bezierControlPointZs.cols))
    {
        return;
    }

    if (doHorizontal)
    {
        computeBezierControlPointsSingleCellHorizontal(controlPointZs, r, c, bezierControlPointZs);
    }
    else
    {
        int gridRows = bezierControlPointZs.rows / 3;

        if (r < gridRows - 1)
        {
            computeBezierControlPointsSingleCellVertical(r, c, bezierControlPointZs);
        }
    }
}

/**
* @brief Kernel to evaluate the b-spline surface at the specified output pixel.
* @param dstMat Output image, contiguous mem, 3 floats per pixel.
* @param gridWidth B-spline grid width, in cells.
* @param gridHeight B-spline grid height, in cells.
* @param zs Control point Z values, contiguous mem.
*/
__global__ void evalBSplineKernel(CudaMat2<CudaPoint3<float>> dstMat, int gridWidth, int gridHeight, CudaMat2<float> controlPointZs, int nPointsPerDim)
{
    // which output pixel this thread is for
    unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if ((yi >= dstMat.rows) || (xi >= dstMat.cols))
    {
        return;
    }

    // which grid cell this pixel is in
    int cellx = xi / nPointsPerDim;
    int celly = yi / nPointsPerDim;

    // indices into zs for this cell
    // 4 control points per cell, with 1 of padding on each side
    int zsx = cellx + 1;
    int zsy = celly + 1;
    int zStride = gridWidth + 3; // 1 on each side, plus 1 because grid width is in cells
    int zsxi = zsy * zStride + zsx; // this is the index of the first coeff for this cell

    // compute the bezier control points
    //float bezierPts[16];

    // evaluate the bezier surface
    float* zs = controlPointZs.dataDevice;
    float x = zs[zsxi];
    float y = zs[zsxi + 1];
    float z = zs[zsxi + zStride];

    //CudaPoint3<float> pt = { x, y, z };
    //CudaPoint3<float> pt1 = { z, y, z };
    //CudaPoint3<float> pt2 = pt * x - pt;


    //// output
    //int idx = yi * dstMat.cols + xi;
    //dstMat.dataDevice[idx] = pt;

    //printf("evalBSplineKernel: xi: %d, yi: %d, idx: %d\n", xi, yi, idx);
}

/**
 * @brief Compute bezier control points from the b-spline control points.
 * This is the version that does horizontal and vertical in separate kernels.
 * There are 3x3 bezier control points per cell.
 * Use one thread per cell.
 * @param deviceId 
 * @param controlPointZs B-spline control points.
 * @param bezierControlPointZs float, 3x3 times the size of controlPointZs.
 * @return Success or not.
 */
bool cudaComputeBezierControlPointsSeparated(int deviceId, CudaMat2<float>& controlPointZs, CudaMat2<float>& bezierControlPointZs)
{
    cudaError_t cudaStatus = cudaSuccess;

    // setting device multiple times has no impact
    setDevice(deviceId);

    // Number of cells.
    // (We include top/left edge points/cells just for simplicity in coding, but they are not computable and not used.)
    int cellRows = controlPointZs.rows;
    int cellCols = controlPointZs.cols;

    // threads per block (16 * 8 == 128 which is number of cuda cores per SM?)
    // TEMP: 1 thread per block for small tests
    dim3 dimBlock(1, 1, 1);

    // however many thread blocks to cover image
    // (if image is not multiple of thread block dims then some will not be computed)
    // We don't process the last column.
    dim3 dimGrid((cellCols - 1) / dimBlock.x, cellRows / dimBlock.y, 1); 

    // dst image
    cudaStatus = bezierControlPointZs.cudaMalloc();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dest image failed!");
        goto Error;
    }

    cudaStatus = controlPointZs.copyHostToDevice(); // also does alloc

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc and copy controlPointZs to device failed!");
        goto Error;
    }

    // horizontal kernel
    // clang-format off
    computeBezierControlPointsKernelSeparated<<<dimGrid, dimBlock, 0>>>(controlPointZs, bezierControlPointZs, true);
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

    // vertical kernel
    // clang-format off
    computeBezierControlPointsKernelSeparated<<<dimGrid, dimBlock, 0>>>(controlPointZs, bezierControlPointZs, false);
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
    cudaStatus = bezierControlPointZs.copyDeviceToHost();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dest image failed!");
        goto Error;
    }

Error:
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
    cudaError_t cudaStatus = cudaSuccess;

    // setting device multiple times has no impact
    setDevice(deviceId);

    // Number of cells.
    // (We include top/left edge points/cells just for simplicity in coding, but they are not computable and not used.)
    int cellRows = controlPointZs.rows;
    int cellCols = controlPointZs.cols;

    // threads per block (16 * 8 == 128 which is number of cuda cores per SM?)
    // TEMP: 1 thread per block for small tests
    dim3 dimBlock(1, 1, 1);

    // however many thread blocks to cover image
    // (if image is not multiple of thread block dims then some will not be computed)
    // We don't process the last column.
    dim3 dimGrid((cellCols - 1) / dimBlock.x, cellRows / dimBlock.y, 1); 

    // dst image
    cudaStatus = bezierControlPointZs.cudaMalloc();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dest image failed!");
        goto Error;
    }

    cudaStatus = controlPointZs.copyHostToDevice(); // also does alloc

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc and copy controlPointZs to device failed!");
        goto Error;
    }

    // horizontal kernel
    // clang-format off
    computeBezierControlPointsKernelSeparated<<<dimGrid, dimBlock, 0>>>(controlPointZs, bezierControlPointZs, true);
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

    // vertical kernel
    // clang-format off
    computeBezierControlPointsKernelSeparated<<<dimGrid, dimBlock, 0>>>(controlPointZs, bezierControlPointZs, false);
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
    cudaStatus = bezierControlPointZs.copyDeviceToHost();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dest image failed!");
        goto Error;
    }

Error:
    bezierControlPointZs.cudaFree();
    controlPointZs.cudaFree();

    return cudaStatus == cudaSuccess;
}

/**
* @brief Evaluate the uniform b-spline surface defined by the control points, with nPointsPerDim * nPointsPerDim per cell.
* @param controlPointZs Control point Z values.
* @param nPointsPerDim How finely to evaluate the surface, in points per dimension per cell.
* @param outputMat Pre-allocated output matrix with (nPointsPerDim * cells) x (nPointsPerDim * cells) matrix of computed values.
* @return Result code.
*/
bool cudaEvalBSpline(int deviceId, CudaMat2<float>& controlPointZs, int nPointsPerDim, CudaMat2<CudaPoint3<float>>& outputMat)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    int gridHeight = controlPointZs.rows - 3; // cells is points - 1, plus 2 for the border
    int gridWidth = controlPointZs.cols - 3;

    int outputHeight = gridHeight * nPointsPerDim;
    int outputWidth = gridWidth * nPointsPerDim;

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

    cudaStatus = controlPointZs.copyHostToDevice(); // also alloc

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc and copy controlPointZs to device failed!");
        goto Error;
    }

    // kernel
    // clang-format off
    evalBSplineKernel<<<dimGrid, dimBlock, 0>>>(outputMat, gridWidth, gridHeight, controlPointZs, nPointsPerDim);
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
    controlPointZs.cudaFree();
    
    return cudaStatus == cudaSuccess;
}
