# CudaGaussianBlur

---

# 모듈 : BmpUtile

---

##개요
- bmp 파일을 RGB 버퍼로 변환하거나, RGB 버퍼를 BMP 파일로 저장하는 기능을 제공.

### BMPFileHeader
- BMP 파일의 기본 헤더 ( 14 bytes )

### BMPDIBHeader
- BMP 이미지의 DIB 헤더 ( 40 bytes )

---

## 주요 함수

---

### bool BmpToRgbBuffers ( ... )
- ***목적*** : BMP 파일을 RGB 버퍼로 읽기
- ***입력*** : 파일 경로, RGB 버퍼
- ***출력*** : 성공 여부, 이미지 크기

---

### 디버깅 및 예외 처리
- 메모리 할당 실패 시 로그 출력
- BMP 형식 오류 처리

---

# main.cu 문서

---

## 개요
- 이 파일을 CUDA를 사용하여 BMP 이미지 데이터를 **병렬 처리**하는 프로그램입니다.
- 목표 : RGB 버퍼를 GPU에서 처리하여 출력 이미지를 생성.

---

## 전체 처리 흐름
1. BMP 이미지 로드 ->
2. GPU 메모리 할당 및 데이터 전송 ->
3. CUDA 커널 실행 ( 픽셀 병렬 처리 ) ->
4. 처리 결과 복사 ->
5. BMP 파일로 저장 ->
6. 메모리 해제 및 종료

---

## 주요 기능
1. Bmp 파일 읽기
- **함수** : 'Bmp::BmpToRgbBuffers()'
- **기능** : BMP 파일에서 RGB 버퍼로 변환 ( CPU 에서 실행 )

2. CUDA 커널 호출
- **커널** : "__global__ void HorizontalBlur(...)', "__global__ void VerticalBlur(...)"
- **기능** : GPU 에서 RGB 데이터를 병럴 처리

결과 저장
- **함수** : "Bmp::RgbBuffersToBmp(...)"
- **기능** : 처리된 RGB 버퍼를 BMP 파일로 저장 ( CPU 에서 실행 )

---

## 가우시안 블러 설명
- 가우시안 분포를 기반으로 커널을 사용해 이미지 값을 부드럽게 만든다
- 사용된 1D 커널 : '[0.1, 0.2, 0.4, 0.2, 0.1]'
  - **대칭** : 중심(0.4)을 기준으로 좌우 대칭
  - **정규화** : 가중치 합이 1.0으로, 픽셀 값 보존.
  - **분리 기능** : 수평 / 수직 방향으로 나눠 효울적 계산

---

## 주요 함수 및 커널 설명

### void AddPadding(unsigned char** dstBuffer, unsigned char* srcBuffer, const int paddingSize, size_t& pitch, const unsigned int& width, const unsigned int& height)
- **목적** : GPU 이미지 처리에 최적화된 메모리 할당
- **매개변수** :
  -  'dstBuffer' : device 메모리 버퍼 ( GPU 메모리 )
  -  'srcBuffer' : host 메모리 버퍼, r, g, b 버퍼
  -  'pitch' : 정렬된 메모리의 행 바이트 수
  -  'width', 'height' : 이미지의 너비와 높이
- **특징**
  -  cudaMallocPitch로 패딩 포함 메모리 할당
  -  cudaMemset2D로 패딩 영역을 0으로 초기화
  -  cudaMemcpy2D로 srcBuffer -> dstBuffer 복사
      - 복사시 dstBuffer의 상, 하, 좌, 우 에 (정수)2 크기의 패딩을 넣음

   
### void RemovePadding(unsigned char* dstBuffer, unsigned char* srcBuffer, const int paddingSize, size_t& pitch, const unsigned int& width, const unsigned int& height)
- **목적** : 패딩 영역을 제외하고, 처리된 결과를 **호스트 메모리로 복사**
- **특징** :
  - cudaMemcpy2D를 사용하여 패딩을 제외한 실제 픽셀 데이터를 복사


### __global__ void HorizontalBlur(unsigned char* resultBuffer, unsigned char* inputBuffer, size_t pitch, unsigned int width, unsigned int height)
- **목적** : GPU에서 각 R, G, B 버퍼의 행 픽셀의 블러 처리
- **매개변수** :
    - 'inputBuffer' : r, g, b 의 데이터 버퍼 ( GPU 메모리 )
    - 'resultBuffer' : blur 처리된 r, g, b 데이터 버퍼 ( GPU 메모리 )
    - 'width', 'height' : 이미지의 너비와 높이
- **동작** :
  1. **스레드 인덱스 계산**
      ```cpp
      int padding = KERNEL_RADIUS;
      int x = blockDim.x * blockIdx.x + threadIdx.x + padding;
      int y = blockIdx.y + padding;
      ```
      - padding = 2 : 경계 처리용으로 스레드 인덱스에 패딩 적용
      - blockDim.x * blockIdx.x + threadIdx.x : X 축 전역 위치
  2. **전역 메모리 인덱스** :
     ```cpp
     int globalIdx = x + y * (pitch / sizeof(unsigned char));
     ```
     - pitch로 행별 바이트 수 보정
  3. **공유 메모리 인덱스** :
     ```cpp
     int localIdx = threadIdx.x + KERNEL_RADIUS;
     ```
  4. **공유 메모리 로드** :
     - ***중앙 픽셀*** 로드 :
       ```cpp
       int localIdx = threadIdx.x + KERNEL_RADIUS;
       ```
     - ***좌/우 경계 픽셀***로드 (thread.x < KERNEL_RADIUS)
       ```cpp
       int leftIdx = (x - KERNEL_RADIUS) < 0 ? 0 : (x - KERNEL_RADIUS);
       int rightIdx = (x + blockDim.x) >= (int)width ? (width - 1) : (x + blockDim.x);
       sharedData[localIdx - KERNEL_RADIUS] = inputBuffer[leftIdx + y * (pitch /     sizeof(unsigned char))];
        sharedData[localIdx + blockDim.x] = inputBuffer[rightIdx + y * (pitch / sizeof(unsigned char))];
        ```
        - leftIdx : 왼쪽 픽셀
        - rightIdx : 오른쪽 픽셀
  5. **블러 계산**
    ```cpp
    float sum = 0.0f;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i)
    {
      sum += sharedData[localIdx + i] * d_kernel[i + KERNEL_RADIUS];
    }
    ```
    - 픽셀에 가중치를 곱한 값을 저장한다
      
### __global__ void VerticalBlur(unsigned char* resultBuffer, unsigned char* inputBuffer, size_t pitch, unsigned int width, unsigned int height)
- **목적** : GPU에서 각 R, G, B 버퍼의 열 픽셀의 블러 처리
- **매개변수** :
    - 'inputBuffer' : r, g, b 의 데이터 버퍼 ( GPU 메모리 )
    - 'resultBuffer' : blur 처리된 r, g, b 데이터 버퍼 ( GPU 메모리 )
    - 'width', 'height' : 이미지의 너비와 높이
- **동작** :
  1. **스레드 인덱스 계산**
      ```cpp
      int padding = KERNEL_RADIUS;
      int x = blockDim.x * blockIdx.x + threadIdx.x + padding;
      int y = blockIdx.y + padding;
      ```
      - padding = 2 : 경계 처리용으로 스레드 인덱스에 패딩 적용
      - blockDim.x * blockIdx.x + threadIdx.x : X 축 전역 위치
  2. **전역 메모리 인덱스** :
     ```cpp
     int globalIdx = x + y * (pitch / sizeof(unsigned char));
     ```
     - pitch로 행별 바이트 수 보정
  3. **공유 메모리 인덱스** :
     ```cpp
     int localIdx = threadIdx.x + KERNEL_RADIUS;
     ```
  4. **공유 메모리 로드** :
     - ***중앙 픽셀*** 로드 :
       ```cpp
       int localIdx = threadIdx.x + KERNEL_RADIUS;
       ```
     - ***상/하 경계 픽셀***로드 (thread.x < KERNEL_RADIUS)
       ```cpp
       int topIdx = (y - KERNEL_RADIUS) < 0 ? 0 : (y - KERNEL_RADIUS);
       int bottomIdx = (y + KERNEL_RADIUS) >= (int)height ? (height - 1) : (y + KERNEL_RADIUS);
       sharedData[localIdx + KERNEL_RADIUS] = inputBuffer[x + topIdx * (pitch / sizeof(unsigned char))];
       sharedData[localIdx + 2 * KERNEL_RADIUS] = inputBuffer[x + bottomIdx * (pitch / sizeof(unsigned char))];
        ```
        - topIdx : 위쪽 픽셀
        - bottomIdx : 아래쪽 픽셀
  5. **블러 계산**
    ```cpp
    float sum = 0.0f;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i)
    {
      sum += sharedData[localIdx + i] * d_kernel[i + KERNEL_RADIUS];
    }
    ```
    - 픽셀에 가중치를 곱한 값을 저장한다

---
