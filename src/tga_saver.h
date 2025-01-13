#pragma once
#include <string_view>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <array>

#include "linear_math.h"
#include "handle_error.h"

/**
 * Saves abuffer on the GPU to a tga file
 *
 * The buffer needs to be in the (float x, float y, uchar4 bgra) format
 *
 * Does not support any kind of compression, color maps etc.
 *
 * based on specs found in https://en.wikipedia.org/wiki/Truevision_TGA
 */
inline void saveToTGA(const std::filesystem::path& path, const size_t width, const size_t height, const Vec3f* cudaDataPtr)
{
    std::vector<Vec3f>  dataBuffer(width*height);
    std::vector<uchar3> rgbBuffer(width*height);
    cudaCheckError(cudaMemcpy(dataBuffer.data(), cudaDataPtr, width * height * sizeof(Vec3f), cudaMemcpyDeviceToHost));
    std::ofstream outFile(path, std::ios::out | std::ios::binary);
    std::cout << std::format("Saving output to file \"{}\" \n", std::filesystem::absolute(path).string());

    std::array<uint8_t,18> tgaHeader{};
    /* 00-01 = 0 */
    tgaHeader[2] = 2;
    /* 03-11 = 0 */
    tgaHeader[12] = (255 & width)        ;  // width   low bits
    tgaHeader[13] = (255 & (width >> 8)) ;  // width  high bits
    tgaHeader[14] = (255 & height)       ;  // height  low bits
    tgaHeader[15] = (255 & (height >> 8));  // height high bits
    tgaHeader[16] = 24;
    tgaHeader[17] = 0b00000000;

    outFile.write(reinterpret_cast<char*>(tgaHeader.data()), 18);
    for (int i = 0; i < width*height; ++i) {
        const int x  = static_cast<int>(dataBuffer[i].x);
        const int y  = static_cast<int>(dataBuffer[i].y);
        const uchar4 px = *reinterpret_cast<uchar4*>(&dataBuffer[i].z);
        rgbBuffer[x + y * width].x = px.z;
        rgbBuffer[x + y * width].y = px.y;
        rgbBuffer[x + y * width].z = px.x;
    }
    outFile.write(reinterpret_cast<char*>(rgbBuffer.data()), static_cast<std::streamsize>(width * height * sizeof(uchar3)));
}