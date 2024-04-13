// Separated as in two kernels, one for horizontal and one for vertical.


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

    horizontalBSplineInterpFull(bsp, bspStride, r, c, bSplineControlPointZs.cols, bzp + r * 3 * bzpStride + c * 3);
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
    dim3 dimGrid(cellCols / dimBlock.x, cellRows / dimBlock.y, 1); 

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

