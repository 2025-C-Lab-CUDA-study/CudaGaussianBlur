#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "BmpUtile.h"


#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())


constexpr int BLOCK_SIZE = 256;
constexpr int KERNEL_SIZE = 5;
constexpr int KERNEL_RADIUS = KERNEL_SIZE / 2;


__constant__ float d_kernel[KERNEL_SIZE] = { 0.1f, 0.2f, 0.4f, 0.2f, 0.1f };



__global__ void HorizontalBlur(unsigned char* resultBuffer, unsigned char* inputBuffer, size_t pitch, unsigned int width, unsigned int height)
{
    extern __shared__ unsigned char sharedData[];

    int padding = KERNEL_RADIUS;
    int x = blockDim.x * blockIdx.x + threadIdx.x + padding;
    int y = blockIdx.y + padding;

    if (x < width - padding && y < height - padding)
    {
        int globalIdx = x + y * (pitch / sizeof(unsigned char));
        int localIdx = threadIdx.x + KERNEL_RADIUS;

        sharedData[localIdx] = inputBuffer[globalIdx];

        if (threadIdx.x < KERNEL_RADIUS)
        {
            int leftIdx = (x - KERNEL_RADIUS) < 0 ? 0 : (x - KERNEL_RADIUS);
            int rightIdx = (x + blockDim.x) >= (int)width ? (width - 1) : (x + blockDim.x);
            sharedData[localIdx - KERNEL_RADIUS] = inputBuffer[leftIdx + y * (pitch / sizeof(unsigned char))];
            sharedData[localIdx + blockDim.x] = inputBuffer[rightIdx + y * (pitch / sizeof(unsigned char))];
        }

        __syncthreads();

        float sum = 0.0f;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i)
        {
            sum += sharedData[localIdx + i] * d_kernel[i + KERNEL_RADIUS];
        }

        sum = sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : sum);
        resultBuffer[globalIdx] = static_cast<unsigned char>(sum);
    }
}

__global__ void VerticalBlur(unsigned char* resultBuffer, unsigned char* inputBuffer, size_t pitch, unsigned int width, unsigned int height)
{
    extern __shared__ unsigned char sharedData[];

    int padding = KERNEL_RADIUS;
    int x = blockIdx.x * blockDim.x + threadIdx.x + padding;
    int y = blockIdx.y + padding;

    if (x < width - padding && y < height - padding)
    {
        int globalIdx = x + y * (pitch / sizeof(unsigned char));
        int localIdx = threadIdx.x;

        sharedData[localIdx] = inputBuffer[globalIdx];

        if (threadIdx.x < KERNEL_RADIUS)
        {
            int topIdx = (y - KERNEL_RADIUS) < 0 ? 0 : (y - KERNEL_RADIUS);
            int bottomIdx = (y + KERNEL_RADIUS) >= (int)height ? (height - 1) : (y + KERNEL_RADIUS);
            sharedData[localIdx + KERNEL_RADIUS] = inputBuffer[x + topIdx * (pitch / sizeof(unsigned char))];
            sharedData[localIdx + 2 * KERNEL_RADIUS] = inputBuffer[x + bottomIdx * (pitch / sizeof(unsigned char))];
        }

        __syncthreads();

        float sum = 0.0f;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i)
        {
            int offsetY = y + i;
            offsetY = offsetY < 0 ? 0 : (offsetY >= (int)height ? (height - 1) : offsetY);
            sum += inputBuffer[x + offsetY * (pitch / sizeof(unsigned char))] * d_kernel[i + KERNEL_RADIUS];
        }

        sum = sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : sum);
        resultBuffer[globalIdx] = static_cast<unsigned char>(sum);
    }
}



void AddPadding(unsigned char** dstBuffer, unsigned char* srcBuffer, const int paddingSize, size_t& pitch, const unsigned int& width, const unsigned int& height)
{
    // Set new buffer ===========================================================================================================

    CUDA_CHECK(cudaMallocPitch(dstBuffer, &pitch, sizeof(unsigned char) * (width + paddingSize * 2), height + paddingSize * 2));
    CUDA_CHECK(cudaMemset2D(*dstBuffer, pitch, 0, sizeof(unsigned char) * (width + paddingSize * 2), height + paddingSize * 2));


    // Copy buffer to new buffer =================================================================================================

    CUDA_CHECK(cudaMemcpy2D(
        *dstBuffer + paddingSize * pitch + paddingSize,
        pitch,
        srcBuffer,
        sizeof(unsigned char) * width,
        sizeof(unsigned char) * width,
        height,
        cudaMemcpyHostToDevice
    ));
}

void RemovePadding(unsigned char* dstBuffer, unsigned char* srcBuffer, const int paddingSize, size_t& pitch, const unsigned int& width, const unsigned int& height)
{
    CUDA_CHECK(cudaMemcpy2D(
        dstBuffer,
        sizeof(unsigned char) * (width - paddingSize * 2),
        srcBuffer + paddingSize * pitch + paddingSize,
        pitch,
        sizeof(unsigned char) * (width - paddingSize * 2),
        height - paddingSize * 2,
        cudaMemcpyDeviceToHost
    ));
}



int main(void)
{
    // Set host data ==================================================================================

    int h_width, h_height;
    unsigned char* h_rBuffer = nullptr;
    unsigned char* h_gBuffer = nullptr;
    unsigned char* h_bBuffer = nullptr;

    const char* path = "C:\\Users\\james\\Documents\\2025\\source_code\\lenna.bmp";
    if (!Bmp::BmpToRgbBuffers(path, &h_rBuffer, &h_gBuffer, &h_bBuffer, h_width, h_height))
    {
        std::cerr << "Error: Reading BMP file failed" << std::endl;
        if (h_rBuffer) free(h_rBuffer);
        if (h_gBuffer) free(h_gBuffer);
        if (h_bBuffer) free(h_bBuffer);
        return 1;
    }


    // Set device data ================================================================================

    const int PADDING_SIZE = KERNEL_RADIUS;
    unsigned int d_width = h_width + PADDING_SIZE * 2;
    unsigned int d_height = h_height + PADDING_SIZE * 2;

    size_t d_rPitch, d_gPitch, d_bPitch, d_calPitch;
    unsigned char* d_rBuffer = nullptr, * d_calRBuffer = nullptr;
    unsigned char* d_gBuffer = nullptr, * d_calGBuffer = nullptr;
    unsigned char* d_bBuffer = nullptr, * d_calBBuffer = nullptr;

    AddPadding(&d_rBuffer, h_rBuffer, PADDING_SIZE, d_rPitch, h_width, h_height);
    AddPadding(&d_gBuffer, h_gBuffer, PADDING_SIZE, d_gPitch, h_width, h_height);
    AddPadding(&d_bBuffer, h_bBuffer, PADDING_SIZE, d_bPitch, h_width, h_height);

    CUDA_CHECK(cudaMallocPitch(&d_calRBuffer, &d_calPitch, sizeof(unsigned char) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_calGBuffer, &d_calPitch, sizeof(unsigned char) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_calBBuffer, &d_calPitch, sizeof(unsigned char) * d_width, d_height));

    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim((d_width + BLOCK_SIZE - 1) / BLOCK_SIZE, d_height);
    size_t sharedMemSize = (BLOCK_SIZE + 2 * KERNEL_RADIUS) * sizeof(unsigned char);


    // Execute Kernel blur ===============================================================================

    for (int i = 0; i < 3; ++i)
    {
        // Horizontal Blur ===================================================================================

        HorizontalBlur << <gridDim, blockDim, sharedMemSize >> > (d_calRBuffer, d_rBuffer, d_rPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();
        HorizontalBlur << <gridDim, blockDim, sharedMemSize >> > (d_calGBuffer, d_gBuffer, d_gPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();
        HorizontalBlur << <gridDim, blockDim, sharedMemSize >> > (d_calBBuffer, d_bBuffer, d_bPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();


        // Vertival Blur =====================================================================================

        VerticalBlur << <gridDim, blockDim, sharedMemSize >> > (d_rBuffer, d_calRBuffer, d_calPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();
        VerticalBlur << <gridDim, blockDim, sharedMemSize >> > (d_gBuffer, d_calGBuffer, d_calPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();
        VerticalBlur << <gridDim, blockDim, sharedMemSize >> > (d_bBuffer, d_calBBuffer, d_calPitch, d_width, d_height);
        CUDA_KERNEL_CHECK();
    }


    // Remove padding and memcpy to host =================================================================

    RemovePadding(h_rBuffer, d_rBuffer, PADDING_SIZE, d_rPitch, d_width, h_height);
    RemovePadding(h_gBuffer, d_gBuffer, PADDING_SIZE, d_gPitch, d_width, h_height);
    RemovePadding(h_bBuffer, d_bBuffer, PADDING_SIZE, d_bPitch, d_width, h_height);


    // Make Bmp file =====================================================================================

    const char* outPath = "C:\\Users\\james\\Documents\\2025\\source_code\\blurredLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath, h_rBuffer, h_gBuffer, h_bBuffer, h_width, h_height))
    {
        std::cerr << "Error: Writing BMP file failed" << std::endl;
    }


    // Free Memory =======================================================================================

    free(h_rBuffer);
    free(h_gBuffer);
    free(h_bBuffer);
    CUDA_CHECK(cudaFree(d_rBuffer));
    CUDA_CHECK(cudaFree(d_gBuffer));
    CUDA_CHECK(cudaFree(d_bBuffer));
    CUDA_CHECK(cudaFree(d_calRBuffer));
    CUDA_CHECK(cudaFree(d_calGBuffer));
    CUDA_CHECK(cudaFree(d_calBBuffer));

    return 0;
}