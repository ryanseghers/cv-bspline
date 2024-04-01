#pragma once
#include "cuda_runtime.h"


/**
* @brief Thin 2-d image wrapper, no automatic memory management/ownership.
* This is designed for having buffers on both host and device.
* This does have methods to alloc and free device memory, but not host.
*/
template <typename T>
struct CudaMat2
{
    T* dataHost = nullptr;
    T* dataDevice = nullptr;

    int rows = 0;
    int cols = 0;

    /**
    * @brief Distance between start of two consecutive rows, in elements not bytes.
    * For example, for CudaPoint2 this is in units of sizeof(CudaPoint2).
    */
    int stride = 0;

    CudaMat2() {}

    /**
     * @brief Host-side constructor.
     */
    CudaMat2(T* data, int rows, int cols, int stride) : dataHost(data), rows(rows), cols(cols), stride(stride) {} 
    
    /**
     * @brief Host-side constructor.
     * Cannot use this for non-continuous buffers because this sets stride to width.
     */
    CudaMat2(int rows, int cols) : dataHost(nullptr), rows(rows), cols(cols), stride(cols) {}

    /**
     * @brief See definition of stride.
     */
    CudaMat2(int rows, int cols, int stride) : rows(rows), cols(cols), stride(stride) {}

    int getWidthInBytes() const
    {
        return cols * sizeof(T);
    }

    int getStrideInBytes() const
    {
        return stride * sizeof(T);
    }

    int getSizeInBytes() const
    {
       return rows * stride * sizeof(T);
    }

    /**
     * @brief Accessor for dataHost. Not available in cuda.
     */
    T& at(int row, int col)
    {
        return dataHost[row * stride + col];
    }

    T getHost(int row, int col)
    {
        return dataHost[row * stride + col];
    }

    void setHost(int row, int col, T val)
    {
        dataHost[row * stride + col] = val;
    }

    T getDevice(int row, int col)
    {
        return dataDevice[row * stride + col];
    }

    void setDevice(int row, int col, T val)
    {
        dataDevice[row * stride + col] = val;
    }

    cudaError_t cudaMalloc()
    {
        if (dataDevice != NULL)
        {
            return cudaErrorAlreadyAcquired; // ? anyway at least it's an error
        }

        return ::cudaMalloc((void**)&dataDevice, getSizeInBytes());
    }

    void cudaFree()
    {
        if (dataDevice != nullptr)
        {
            ::cudaFree(dataDevice);
            dataDevice = nullptr;
        }
    }

    cudaError_t copyDeviceToHost()
    {
        return cudaMemcpy2D(
            (void*)dataHost,
            getStrideInBytes(),
            (void*)dataDevice,
            getStrideInBytes(),
            getWidthInBytes(),
            rows,
            cudaMemcpyDeviceToHost);
    }

    /**
     * @brief This will alloc if needed.
     */
    cudaError_t copyHostToDevice()
    {
        cudaError_t status = cudaSuccess;

        if (dataDevice == nullptr)
        {
            status = cudaMalloc();

            if (status != cudaSuccess)
            {
                return status;
            }
        }

        return cudaMemcpy2D(
            (void*)dataDevice,
            getStrideInBytes(),
            (void*)dataHost,
            getStrideInBytes(),
            getWidthInBytes(),
            rows,
            cudaMemcpyHostToDevice);
    }
};

template <typename T>
struct CudaPoint2
{
    T x;
    T y;
};

template <typename T>
struct CudaPoint3
{
    T x;
    T y;
    T z;

    // CUDA doesn't like this
    //CudaPoint3() {}
    //CudaPoint3(T x, T y, T z) : x(x), y(y), z(z) {}

    //CudaPoint3<T> operator* (T scalar) const
    //{
    //    return CudaPoint3(x * scalar, y * scalar, z * scalar);
    //}

    //CudaPoint3<T> operator/ (T scalar) const
    //{
    //    return CudaPoint3(x / scalar, y / scalar, z / scalar);
    //}

    //CudaPoint3<T> operator+ (CudaPoint3<T> other) const
    //{
    //    return CudaPoint3(x + other.x, y + other.y, z + other.z);
    //}

    //CudaPoint3<T> operator- (CudaPoint3<T> other) const
    //{
    //    return CudaPoint3(x - other.x, y - other.y, z - other.z);
    //}

    //CudaPoint3<T> operator+= (CudaPoint3<T> other) const
    //{
    //    x += other.x;
    //    y += other.y;
    //    z += other.z;
    //    return *this;
    //}

    //CudaPoint3<T> operator-= (CudaPoint3<T> other) const
    //{
    //    x -= other.x;
    //    y -= other.y;
    //    z -= other.z;
    //    return *this;
    //}

    //CudaPoint3<T> operator*= (T scalar) const
    //{
    //    x *= scalar;
    //    y *= scalar;
    //    z *= scalar;
    //    return *this;
    //}

    //CudaPoint3<T> operator/= (T scalar) const
    //{
    //    x /= scalar;
    //    y /= scalar;
    //    z /= scalar;
    //    return *this;
    //}
};

/**
* @brief Linear interpolation between two values or points.
*/
template <typename T>
__device__ T interpolate(T p0, T p1, float t)
{
    return p0 + (p1 - p0) * t;
}

//__device__ float interpolate(float p0, float p1, float t)
//{
//    return p0 + (p1 - p0) * t;
//}

void assertCudaStatus(cudaError_t cudaStatus, const char* msg);
void setDevice(int deviceId = 0);
void printDeviceInfo(int deviceId = 0);
void setupSrcImageTexture16u(int width, int height, const uint16_t* psrc, int samplingType, cudaArray_t& srcArray, cudaTextureObject_t& srcTexture);
