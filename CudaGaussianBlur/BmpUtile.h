#pragma once
#include <cstdint>

// --- BMP 파일 헤더 (14바이트) ---
#pragma pack(push, 1)
struct BMPFileHeader
{
    uint16_t bfType;      // 'BM'
    uint32_t bfSize;      // 파일 크기
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;   // 픽셀 데이터 시작 위치
};

// --- BMP DIB 헤더 (BITMAPINFOHEADER, 40바이트) ---
struct BMPDIBHeader
{
    uint32_t biSize;          // DIB 헤더 크기 (40)
    int32_t  biWidth;         // 너비 (픽셀)
    int32_t  biHeight;        // 높이 (픽셀)
    uint16_t biPlanes;        // 항상 1
    uint16_t biBitCount;      // 비트 깊이 (24bpp)
    uint32_t biCompression;   // 압축 (0=BI_RGB)
    uint32_t biSizeImage;     // 이미지 크기 (0 or 실제)
    int32_t  biXPelsPerMeter; // 수평 해상도
    int32_t  biYPelsPerMeter; // 수직 해상도
    uint32_t biClrUsed;       // 색상 수 (0=전부)
    uint32_t biClrImportant;  // 중요 색상 수 (0=전부)
};
#pragma pack(pop)

namespace Bmp
{
    bool BmpToRgbBuffers(const char* filePath, unsigned char** rBuf, unsigned char** gBuf, unsigned char** bBuf, int& width, int& height);
    bool RgbBuffersToBmp(const char* outputPath, const unsigned char* rBuf, const unsigned char* gBuf, const unsigned char* bBuf, const int& width, const int& height);
}
