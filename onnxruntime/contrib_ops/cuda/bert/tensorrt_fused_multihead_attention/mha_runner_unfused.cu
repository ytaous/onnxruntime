/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include "mha_runner.h"
#include <cublas_v2.h>
#include <cub/cub.cuh>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template <typename T>
__device__ inline T myExp(const T x);

template <>
__device__ inline float myExp<float>(const float x)
{
    return __expf(x);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount,
    cublasGemmAlgo_t algo);

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{

    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUDA_R_32F, algo);
}

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{
    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &beta, C, CUDA_R_16F, ldc, strideC, batchCount, CUDA_R_16F, algo);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount);

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount)
{

    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount)
{
    return cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

struct CublasConfigHelper
{
    cublasPointerMode_t pm;
    cublasMath_t mm;
    cublasHandle_t cublas;
    CublasConfigHelper(cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        CUBLAS_CALL(cublasGetPointerMode(cublas, &pm));
        CUBLAS_CALL(cublasGetMathMode(cublas, &mm));
        CUBLAS_CALL(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_CALL(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
    }
    ~CublasConfigHelper()
    {
        cublasSetMathMode(cublas, mm);
        cublasSetPointerMode(cublas, pm);
    }
};

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmaxSmall(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{

    using BlockReduce = cub::BlockReduce<float, TPB>;

    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;
    __shared__ float fMax;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(-FLT_MAX);

    const int idx = offset + threadIdx.x;
    if (threadIdx.x < lastValid)
    {
        threadData = input[idx];
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        fMax = maxElem;
    }
    __syncthreads();

    if (threadIdx.x < lastValid)
    {
        threadData = exp((threadData - fMax) * w);
    }
    else
    {
        threadData = 0;
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = (1.f) / Z;
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        // this will be 0 for threadIdx.x >= lastValid
        output[idx] = T(threadData * rZ);
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmax(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;
    __shared__ float fMax;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(-FLT_MAX);

    if (lastValid >= blockDim.x)
    {
        threadData = 0;
    }
    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        threadData = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        fMax = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        threadData += exp((static_cast<float>(input[idx]) - fMax) * w);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const float val = (i < lastValid) ? exp((static_cast<float>(input[idx]) - fMax) * w) * rZ : 0.f;
        output[idx] = T(val);
    }
}

template <typename T, int TPB, int VPT>
__global__ void maskedSoftmax(const float rsqrtHeadSize, const T* input, T* output, const int* maskIdx)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;

    union SMem
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
        SMem() {}
    };
    __shared__ SMem tmp;

    // grid: (NxS, B)
    const int b = blockIdx.y;
    const int blockOffset = (b * gridDim.x + blockIdx.x) * TPB;
    __shared__ int lastValid;
    if (threadIdx.x == 0)
    {
        lastValid = min(TPB, maskIdx[b]);
    }
    __syncthreads();
    float local[VPT];

    __shared__ float rZ;
    __shared__ float fMax[VPT];

    const int idx = (blockOffset + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = (threadIdx.x < lastValid) ? float(tmp.shm[it * TPB + threadIdx.x]) : -FLT_MAX;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
        if (threadIdx.x == 0)
        {
            fMax[it] = maxElem;
        }
        __syncthreads();
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = (threadIdx.x < lastValid) ? myExp<float>(rsqrtHeadSize * (local[it] - fMax[it])) : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

        if (threadIdx.x == 0)
        {
            rZ = (1.f) / Z;
        }
        __syncthreads();
        local[it] *= rZ;
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        tmp.shm[it * TPB + threadIdx.x] = local[it];
    }
    __syncthreads();
    copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, int TPB, int VPT>
__global__ void softmax(const float rsqrtHeadSize, const T* input, T* output)
{
    float local[VPT];

    using BlockReduce = cub::BlockReduce<float, TPB>;

    union SMem
    {
        T shm[VPT * TPB];
        typename BlockReduce::TempStorage reduce;
        SMem() {}
    };
    __shared__ SMem tmp;

    __shared__ float rZ;
    __shared__ float fMax[VPT];

    const int idx = (TPB * blockIdx.x + threadIdx.x) * VPT;
    T* myshm = &tmp.shm[threadIdx.x * VPT];
    copy<sizeof(T) * VPT>(&input[idx], myshm);

    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = float(tmp.shm[it * TPB + threadIdx.x]);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
        if (threadIdx.x == 0)
        {
            fMax[it] = maxElem;
        }
        __syncthreads();
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        local[it] = myExp<float>(rsqrtHeadSize * (local[it] - fMax[it]));
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {

        const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

        if (threadIdx.x == 0)
        {
            rZ = 1.f / Z;
        }
        __syncthreads();
        local[it] *= rZ;
    }

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        tmp.shm[it * TPB + threadIdx.x] = local[it];
    }
    __syncthreads();
    copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, unsigned TPB>
__global__ void scaledSoftmaxKernelSmall(const int ld, const float rsqrtHeadSize, const T* input, T* output)
{
    scaledSoftmaxSmall<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void scaledSoftmaxKernel(const int ld, const float rsqrtHeadSize, const T* input, T* output)
{
    scaledSoftmax<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T>
int computeScaledSoftmax(
    cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize, const T* input, T* output)
{

    constexpr int VPT = 16 / sizeof(T);

    const dim3 grid(ld * N, B, 1);

    if (ld <= 32)
    {
        const int blockSize = 32;
        scaledSoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }
    else if (ld < 128)
    {
        const int blockSize = 128;
        scaledSoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }
    else if (ld == 128)
    {
        const int grid = B * N * ld / (VPT);
        softmax<T, 128, VPT><<<grid, 128, 0, stream>>>(rsqrtHeadSize, input, output);
    }

    else if (ld == 384)
    {
        const int grid = B * N * ld / (VPT);
        softmax<T, 384, VPT><<<grid, 384, 0, stream>>>(rsqrtHeadSize, input, output);
    }
    else
    {
        const int blockSize = 256;

        scaledSoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
    }

    CUDA_CALL_THROW(cudaPeekAtLastError());
    return 0;
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernelSmall(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output)
{
    __shared__ int lastValid;

    if (threadIdx.x == 0)
    {
        lastValid = min(ld, maskIdx[blockIdx.y]);
    }
    __syncthreads();

    scaledSoftmaxSmall<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernel(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output)
{

    __shared__ int lastValid;

    if (threadIdx.x == 0)
    {
        lastValid = min(ld, maskIdx[blockIdx.y]);
    }
    __syncthreads();
    scaledSoftmax<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T>
int computeMaskedScaledSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize,
    const int* maskIdx, const T* input, T* output)
{
    // Mask idx is of length B and assumes the valid region is contiguous starting
    // from the beginning of the sequence

    const dim3 grid(ld * N, B, 1);
    // for smaller problems, e.g. BERT base B=1, this is not optimal
    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        maskedScaledSoftmaxKernelSmall<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }
    else if (ld < 128)
    {
        constexpr int blockSize = 128;
        maskedScaledSoftmaxKernelSmall<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }
    else if (ld == 128)
    {
        if (B == 1)
        {
            constexpr int VPT = 4 / sizeof(T);
            constexpr int blockSize = 128;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
        else
        {
            constexpr int VPT = 16 / sizeof(T);
            constexpr int blockSize = 128;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
    }
    else if (ld == 384)
    {
        if (B == 1)
        {
            constexpr int VPT = 4 / sizeof(T);
            constexpr int blockSize = 384;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
        else
        {
            constexpr int VPT = 16 / sizeof(T);
            constexpr int blockSize = 384;
            const dim3 grid(ld * N / VPT, B, 1);
            maskedSoftmax<T, blockSize, VPT><<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
        }
    }
    else
    {
        constexpr int blockSize = 256;
        maskedScaledSoftmaxKernel<T, blockSize>
            <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
    }

    CUDA_CALL_THROW(cudaPeekAtLastError());
    return 0;
}

std::pair<int, int> tuneBatchedGemm(
    const int B, const int S, const int numHeads, const int headSize, const int smVersion)
{
    const int nruns = 500;
    cublasHandle_t cublas;
    CUBLAS_CALL_THROW(cublasCreate(&cublas));
    cudaStream_t stream;
    CUDA_CALL_THROW(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CALL_THROW(cudaEventCreate(&start));
    CUDA_CALL_THROW(cudaEventCreate(&stop));
    CUBLAS_CALL_THROW(cublasSetStream(cublas, stream));
    CUBLAS_CALL_THROW(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    using T = half;
    const int omatSize = S * S;
    const int numMats = B * numHeads;
    const int ldQKV = 3 * B * numHeads * headSize;
    const int strideQKV = 3 * headSize;
    const int ldOut = B * numHeads * headSize;
    const int strideOut = headSize;

    const size_t inBytes = S * B * 3 * numHeads * headSize * sizeof(T);
    const size_t qkBytes = S * S * B * numHeads * sizeof(T);
    const size_t outBytes = S * B * numHeads * headSize * sizeof(T);

    T* input = nullptr;
    T* qkptr = nullptr;
    T* output = nullptr;
    CUDA_CALL_THROW(cudaMalloc(&input, inBytes));
    CUDA_CALL_THROW(cudaMalloc(&qkptr, qkBytes));
    CUDA_CALL_THROW(cudaMalloc(&output, outBytes));
    CUDA_CALL_THROW(cudaMemset(input, 1, inBytes));
    CUDA_CALL_THROW(cudaMemset(qkptr, 1, qkBytes));

    // input: SxBx3xNxH
    const T* qptr = input;
    const T* kptr = qptr + headSize;
    const T* vptr = kptr + headSize;

    const int startAlgo = (int) CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    const int endAlgo = (int) CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int best1 = startAlgo;
    int best2 = startAlgo;
    float ms1 = 1000000;
    float ms2 = 1000000;

    ORT_ENFORCE(smVersion >= kSM_53);
    for (int a = startAlgo; a <= endAlgo; a++)
    {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(a);
        float ms1_, ms2_;
        // qkptr: BxNxSxS
        CUDA_CALL_THROW(cudaEventRecord(start, stream));
        for (int r = 0; r < nruns; r++)
        {
            CUBLAS_CALL_THROW(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, headSize, T(1.f),
                kptr, ldQKV, strideQKV, qptr, ldQKV, strideQKV, T(0.f), qkptr, S, omatSize, numMats, algo));
        }

        CUDA_CALL_THROW(cudaEventRecord(stop, stream));
        CUDA_CALL_THROW(cudaStreamSynchronize(stream));
        CUDA_CALL_THROW(cudaEventElapsedTime(&ms1_, start, stop));
        if (ms1_ < ms1)
        {
            best1 = algo;
            ms1 = ms1_;
        }

        // pptr: BxNxSxS
        // output: SxBxNxH
        CUDA_CALL_THROW(cudaEventRecord(start, stream));
        for (int r = 0; r < nruns; r++)
        {
            CUBLAS_CALL_THROW(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, S, S, 1.f,
                vptr, ldQKV, strideQKV, qkptr, S, omatSize, 0.f, output, ldOut, strideOut, numMats, algo));
        }

        CUDA_CALL_THROW(cudaEventRecord(stop, stream));
        CUDA_CALL_THROW(cudaStreamSynchronize(stream));
        CUDA_CALL_THROW(cudaEventElapsedTime(&ms2_, start, stop));

        if (ms2_ < ms2)
        {
            best2 = algo;
            ms2 = ms2_;
        }
    }

    CUDA_CALL_THROW(cudaFree(input));
    CUDA_CALL_THROW(cudaFree(qkptr));
    CUDA_CALL_THROW(cudaFree(output));
    CUDA_CALL_THROW(cudaEventDestroy(start));
    CUDA_CALL_THROW(cudaEventDestroy(stop));
    CUDA_CALL_THROW(cudaStreamDestroy(stream));
    CUBLAS_CALL_THROW(cublasDestroy(cublas));
    return std::make_pair(best1, best2);
}

template int computeScaledSoftmax<float>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const float* input, float* output);
template int computeScaledSoftmax<half>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const half* input, half* output);

template int computeMaskedScaledSoftmax<float>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const int* maskIdx, const float* input, float* output);
template int computeMaskedScaledSoftmax<half>(cudaStream_t stream, const int ld, const int B, const int N,
    const float rsqrtHeadSize, const int* maskIdx, const half* input, half* output);

UnfusedMHARunnerFp16::UnfusedMHARunnerFp16(const int numHeads, const int headSize, const int sm)
    : MHARunner(numHeads, headSize, sizeof(half))
    , mIsBestAlgoFound(false)
    , mAlgoBatchedEx1(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mAlgoBatchedEx2(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mSm(sm)
{
    CUBLAS_CALL_THROW(cublasCreate(&mCublas));
}

UnfusedMHARunnerFp16::~UnfusedMHARunnerFp16()
{
    CUBLAS_CALL_THROW(cublasDestroy(mCublas));
}

void UnfusedMHARunnerFp16::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    if (!mIsBestAlgoFound)
    {
        std::tie(mAlgoBatchedEx1, mAlgoBatchedEx2) = tuneBatchedGemm(B, S, mNumHeads, mHeadSize, mSm);
        mIsBestAlgoFound = true;
        // BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 1 for batch gemms: ", mAlgoBatchedEx1);
        // BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 2 for batch gemms: ", mAlgoBatchedEx2);
    }

}

size_t UnfusedMHARunnerFp16::getWorkspaceSize() const
{
    return 2UL * mWordSize * mOmatSize * mNumMats;
}


void UnfusedMHARunnerFp16::run(const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    const int* maskIdx = static_cast<const int*>(maskPtr);

    CUBLAS_CALL_THROW(cublasSetStream(mCublas, stream));

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)

    CublasConfigHelper helper(mCublas);
    const half* qptr = static_cast<const half*>(qkvPtr);
    const half* kptr = qptr + mHeadSize;
    const half* vptr = kptr + mHeadSize;
    half* qkptr = static_cast<half*>(workspace);
    half* pptr = qkptr + mOmatSize * mNumMats;
    half alpha = 1.f;
    half beta = 0.f;
    CUBLAS_CALL_THROW(::cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mS, mS, mHeadSize, &alpha,
                                                   kptr, CUDA_R_16F, mLdQKV, mStrideQKV, qptr, CUDA_R_16F, mLdQKV, mStrideQKV, &beta, qkptr, CUDA_R_16F, mS,
                                                   mOmatSize, mNumMats, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx1)));

    // apply softmax
    if (maskIdx) {  // if we have a mask
      computeMaskedScaledSoftmax<half>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, maskIdx, qkptr, pptr);
    } else {  // if we don't have a mask
      computeScaledSoftmax<half>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, qkptr, pptr);
    }

    // compute P*V (as V*P)
    CUBLAS_CALL_THROW(cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, mS, mS, &alpha,
                                                 vptr, CUDA_R_16F, mLdQKV, mStrideQKV, pptr, CUDA_R_16F, mS, mOmatSize, &beta, output, CUDA_R_16F, mLdOut,
                                                 mStrideOut, mNumMats, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx2)));
}

void UnfusedMHARunnerFp16::setScaleList(const float scaleQkv, const float scaleCtx, const float dqProbs)
{
}

bool UnfusedMHARunnerFp16::isValid(int s) const
{
    return true;
}

UnfusedMHARunnerFp32::UnfusedMHARunnerFp32(const int numHeads, const int headSize, const int sm)
    : MHARunner(numHeads, headSize, sizeof(float))
    , mIsBestAlgoFound(false)
    , mAlgoBatchedEx1(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mAlgoBatchedEx2(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    , mSm(sm)
{
    CUBLAS_CALL_THROW(cublasCreate(&mCublas));
}

UnfusedMHARunnerFp32::~UnfusedMHARunnerFp32()
{
    CUBLAS_CALL_THROW(cublasDestroy(mCublas));
}

void UnfusedMHARunnerFp32::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
}

size_t UnfusedMHARunnerFp32::getWorkspaceSize() const
{
    return 2UL * mWordSize * mOmatSize * mNumMats;
}


void UnfusedMHARunnerFp32::run(const void* qkvPtr, const void* maskPtr, void* output, void* workspace, cudaStream_t stream)
{
    const int* maskIdx = static_cast<const int*>(maskPtr);

    CUBLAS_CALL_THROW(cublasSetStream(mCublas, stream));

    // Q, K, V: BxNxSxH (inputs)
    // Q * K': BxNxSxS (-> scratch1)
    // P: BxNxSxS (-> scratch2)
    // P * V: BxNxSxH (output)
    const float* qptr = static_cast<const float*>(qkvPtr);
    const float* kptr = qptr + mHeadSize;
    const float* vptr = kptr + mHeadSize;
    float* qkptr = static_cast<float*>(workspace);
    float* pptr = qkptr + mOmatSize * mNumMats;
    float* outptr = static_cast<float*>(output);
    CUBLAS_CALL_THROW(cublasGemmStridedBatched<float>(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mS, mS, mHeadSize, 1.f,
        kptr, mLdQKV, mStrideQKV, qptr, mLdQKV, mStrideQKV, 0.f, qkptr, mS, mOmatSize, mNumMats));

    // apply softmax
    if (maskIdx)
    { // if we have a mask
        computeMaskedScaledSoftmax<float>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, maskIdx, qkptr, pptr);
    }
    else
    { // if we don't have a mask
        computeScaledSoftmax<float>(stream, mS, mB, mNumHeads, mRsqrtHeadSize, qkptr, pptr);
    }

    CUBLAS_CALL_THROW(cublasGemmStridedBatched<float>(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, mS, mS, 1.f,
        vptr, mLdQKV, mStrideQKV, pptr, mS, mOmatSize, 0.f, outptr, mLdOut, mStrideOut, mNumMats));
}

void UnfusedMHARunnerFp32::setScaleList(const float scaleQkv, const float scaleCtx, const float dqProbs)
{
}

bool UnfusedMHARunnerFp32::isValid(int s) const
{
    return true;
}


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
