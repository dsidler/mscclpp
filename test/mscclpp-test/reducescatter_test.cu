// Copyright (c) Microsoft Corporation.
// Copyright (c) Advanced Micro Devices, Inc.
// Licensed under the MIT license.

#include <algorithm>
#include <cstring>
#include <mscclpp/concurrency_device.hpp>
#include <string>

#include "common.hpp"

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

constexpr int NRANKS = 8;
constexpr int NPEERS = NRANKS - 1;
constexpr int CHUNK_SIZE_KB = 256;
constexpr int MAX_BLOCKS = 56;
const size_t SCRATCH_BUFF_SIZE = CHUNK_SIZE_KB * 1024 * MAX_BLOCKS * NRANKS * 2;

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[512];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemOutOfPlaceChans[512];

static void* inputBuff = nullptr;
static void* scratchBuff = nullptr;
static void* outputBuff = nullptr;

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T clip(T val) {
  return val;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return clip(a + b);
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

// Pull-based ReduceScatter implementation
// Code from ALlReduce8Read reused
// Assumes out-of-place
__global__ void __launch_bounds__(512, 1)
    reducescatter5(int* buff, int* resultBuff, size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode,
                   size_t nelems) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = constMemChans + chanOffset;

  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
      (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * CHUNK_SIZE_KB / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }

  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;

  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
  __syncthreads();

  for (size_t i = 0; i < nItrs; ++i) {
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx];
  #pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        int4 val = memChans[peerIdx].read<int4>(nInt4PerRank * rank + offsetOfThisBlock + idx);
        data = add_vectors<int>(val, data);
      }
      resultBuff4[offsetOfThisBlock + idx] = data;
    }
    offsetOfThisBlock += nInt4PerChunk;
  }

  if (restNInt4 > 0) {
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx];
#pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        int4 val = memChans[peerIdx].read<int4>(nInt4PerRank * rank + offsetOfThisBlock + idx);
        data = add_vectors<int>(val, data);
      }
      resultBuff4[offsetOfThisBlock + idx] = data;
    }
  }

  __syncthreads();
  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    memChans[threadIdx.x].signal();
    memChans[threadIdx.x].wait();
  }
}

// Push-based ReduceScatter implementation
// On push, every warp stores to a different peer
// This implementation doesn't provide any benefit and was implemented as an experiment
// Inspired by AllGather6 implementation
// Note incorrect for very small sizes (<=32KB)
// Assumes out-of-place
__global__ void __launch_bounds__(1024, 1)
    reducescatter6(int* buff, int* scratch, int* resultBuff, size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelems) {
  const size_t nBlock = gridDim.x;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;
  const size_t myWid = threadIdx.x / WARP_SIZE;

  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto scratchChans = constMemOutOfPlaceChans + chanOffset;

  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);


     // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
    (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * CHUNK_SIZE_KB / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }

  uint32_t prodFlag = 0;
  uint32_t conFlag = 0;
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t chunkSizePerRank = nNeededBlocks * nInt4PerChunk;
  const size_t blockOffset = nInt4PerChunk * blockIdx.x;
  size_t scratchBaseOffsetInt4 = 0;


  const size_t UNROLL = 8;
  const size_t unitBytesPerThread = 16 * UNROLL;
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t nWarpPerBlock = blockDim.x / WARP_SIZE;
  const size_t nWarpPerChunk = nInt4PerChunk * sizeof(int4) / unitBytesPerWarp;
  const size_t nWarpRest = restNInt4 * sizeof(int4) / unitBytesPerWarp;

  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    scratchChans[threadIdx.x].relaxedSignal();
    scratchChans[threadIdx.x].wait();
  }
  __syncthreads();


  // pre-push
  size_t nWarpsPrePush = (nItrs != 0) ? nWarpPerChunk : nWarpRest;
  nWarpsPrePush *= nPeer;
  size_t myLid = threadIdx.x % WARP_SIZE;
  for (size_t i = myWid;  i < nWarpsPrePush; i += nWarpPerBlock) {
    const size_t gWid = i;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;

    #pragma unroll
    for (size_t j = 0; j < UNROLL; j++) {
      int4 val = buff4[nInt4PerRank * remoteRank + offsetOfThisBlock + (gWid / nPeer) * (WARP_SIZE * UNROLL) + (j * WARP_SIZE) + myLid];
      scratchChans[peerIdx].write(scratchBaseOffsetInt4 + (nInt4PerChunk * rank) + (gWid / nPeer) * (WARP_SIZE * UNROLL) + (j * WARP_SIZE) + myLid, val);
    }
  }
  prodFlag = !prodFlag;
  __syncthreads();
  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    scratchChans[threadIdx.x].signal();
  }

  size_t nextOffsetOfThisBlock = offsetOfThisBlock + nInt4PerChunk;
  for (size_t i = 0; i < nItrs; ++i) {
    // Write to remote peers
    size_t nWarpNextPush = (i+1 == nItrs) ? nWarpRest : nWarpPerChunk;
    if (nWarpNextPush != 0) {
      scratchBaseOffsetInt4 = (prodFlag) ? (worldSize * nInt4PerChunk) : 0;
      nWarpNextPush *= nPeer;

      for (size_t j = myWid;  j < nWarpNextPush; j += nWarpPerBlock) {
        const size_t gWid = j;
        const size_t peerIdx = gWid % nPeer;
        const size_t remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;

        #pragma unroll
        for (size_t j = 0; j < UNROLL; j++) {
          int4 val = buff4[nInt4PerRank * remoteRank + offsetOfThisBlock + (gWid / nPeer) * (WARP_SIZE * UNROLL) + (j * WARP_SIZE) + myLid];
          scratchChans[peerIdx].write(scratchBaseOffsetInt4 + (nInt4PerChunk * rank) + (gWid / nPeer) * (WARP_SIZE * UNROLL) + (j * WARP_SIZE) + myLid, val);
        }

      }
    }

    //Syncrhonize writes to scratch
      prodFlag = !prodFlag;
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      scratchChans[threadIdx.x].wait();
    }
    __syncthreads();
    if (nWarpNextPush != 0) {
      if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
        scratchChans[threadIdx.x].signal();
      }
    }

    // Read from scratch, reduce, and write to result
    scratchBaseOffsetInt4 = (conFlag) ? (worldSize * nInt4PerChunk) : 0;
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx];
      #pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[scratchBaseOffsetInt4 + (nInt4PerChunk * remoteRank) + idx];
        data = add_vectors<int>(val, data);
      }
      resultBuff4[offsetOfThisBlock + idx] = data;
    }
     offsetOfThisBlock += nInt4PerChunk;
     nextOffsetOfThisBlock += nInt4PerChunk;
     conFlag = !conFlag;
  }

  if (restNInt4 > 0) {

    //Syncrhonize writes to scratch
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      scratchChans[threadIdx.x].wait();
    }
    __syncthreads();

    scratchBaseOffsetInt4 = (conFlag) ? (worldSize * nInt4PerChunk) : 0;
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx];
      #pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[scratchBaseOffsetInt4 + (nInt4PerChunk * remoteRank) + idx];
        data = add_vectors<int>(val, data);
      }
      resultBuff4[offsetOfThisBlock + idx] = data;
    }
  }

    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      scratchChans[threadIdx.x].signal();
      scratchChans[threadIdx.x].wait();
    }

}

// Push-based implementation
// Similar to AllReduce8 but with the aim to allow
// for more overlap between chunks by pushing two chunks
// before reading  and reducing one
// Assumes out-of-place
__global__ void __launch_bounds__(1024, 1)
    reducescatter7(int* buff, int* scratch, int* resultBuff, size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelems) {
  const size_t nBlock = gridDim.x;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto scratchChans = constMemOutOfPlaceChans + chanOffset;

  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);


     // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
    (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * CHUNK_SIZE_KB / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }
  
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t scratchBaseOffsetInt4 = blockIdx.x * nInt4PerChunk * worldSize;

  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    scratchChans[threadIdx.x].relaxedSignal();
    scratchChans[threadIdx.x].wait();
  }

  const size_t nChunksToProcess = nItrs + (restNInt4 != 0);
  for (size_t i = 0; i < nChunksToProcess; ++i) {
    __syncthreads();
    // Write to remote peers

    const size_t nInt4ThisItr = (i == nItrs) ? restNInt4 : nInt4PerChunk;
    for (size_t idx = threadIdx.x; idx < nInt4ThisItr; idx += blockDim.x) {
      #pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + offsetOfThisBlock + idx];
        scratchChans[peerIdx].write(scratchBaseOffsetInt4 + (nInt4PerChunk * rank) + idx, val);
      }
    }

    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      scratchChans[threadIdx.x].signal();
      scratchChans[threadIdx.x].wait();
    }
    __syncthreads();

    // Read from scratch, reduce, and write to result
    for (size_t idx = threadIdx.x; idx < nInt4ThisItr; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx];
      #pragma unroll
      for (size_t peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[scratchBaseOffsetInt4 + (nInt4PerChunk * remoteRank) + idx];
        data = add_vectors<int>(val, data);
      }
      resultBuff4[offsetOfThisBlock + idx] = data;
    }
    offsetOfThisBlock += nInt4PerChunk;

    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      scratchChans[threadIdx.x].signal();
      scratchChans[threadIdx.x].wait();
    }
  }
}

// Experimental pull-based implementation
// Threads in the warp are split into 8 groups (== #ranks)
// Each group loads data from a different a different peer
// Then warp-level shufl instruction is used to shift data cross group and reduce it
// Note: this is implementation doesn't perform well
// Assumes out-of-place
__global__ void __launch_bounds__(512, 1)
    reducescatter9(int* buff, int* resultBuff, size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode,
                   size_t nelems) {
  const size_t nBlock = gridDim.x;
  const size_t nWarpsPerBlock = blockDim.x / WARP_SIZE;

  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = constMemChans + chanOffset;

  const size_t warpId = threadIdx.x / WARP_SIZE;
  if (warpId > 8) return;
  const size_t tidInWarp = threadIdx.x % WARP_SIZE;
  const size_t blockId = blockIdx.x;

  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
      (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * CHUNK_SIZE_KB / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }

  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;

  const size_t nBuffers = 7;

  __shared__ int4 localBuffer[WARP_SIZE * 8][nBuffers];
  __shared__ uint64_t producedCount[8];
  __shared__ uint64_t consumedCount;

  if (threadIdx.x < static_cast<uint32_t>(1)){
    consumedCount = 0;
  }
  if (threadIdx.x < static_cast<uint32_t>(8)) {
    producedCount[threadIdx.x] = 0;
  }

  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t UNROLL = 4;

  for (size_t i = 0; i < nItrs; ++i) {
    for (size_t idx = 0; idx < nInt4PerChunk; idx += (UNROLL * WARP_SIZE / 8)) {
      int4 data_arr[UNROLL];
      size_t peerIdx = tidInWarp / 8;
      size_t int4Idx = tidInWarp % 8;

      // load multiple
      if (peerIdx != 7) {
        #pragma UNROLL
        for (size_t u = 0; u < UNROLL; u++) {
          data_arr[u] = memChans[peerIdx].read<int4>(nInt4PerRank * rank + offsetOfThisBlock + idx + int4Idx + (u * (WARP_SIZE / 8)));
        }
      } else {
        #pragma UNROLL
        for (size_t u = 0; u < UNROLL; u++) {
          data_arr[u] = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx + int4Idx + (u * (WARP_SIZE / 8))];
        }
      }
      // reduce and store
      #pragma UNROLL
      for (size_t u = 0; u < UNROLL; u++) {
        // Reduce to peerIdx 0
        for (int delta = (WARP_SIZE / 2); delta > 4; delta >>= 1) {
          int4 val;
          uint64_t mask = 0x101010101010101;
          val.x = __shfl_down_sync(mask, data_arr[u].x, delta);
          val.y = __shfl_down_sync(mask, data_arr[u].y, delta);
          val.z = __shfl_down_sync(mask, data_arr[u].z, delta);
          val.w = __shfl_down_sync(mask, data_arr[u].w, delta);
          data_arr[u] = add_vectors<int>(val, data_arr[u]);
        }

        if (tidInWarp < (WARP_SIZE / 8)) {
          resultBuff4[offsetOfThisBlock + idx + int4Idx + (u * (WARP_SIZE/8))] = data_arr[u];
        }
      }
    }
    offsetOfThisBlock += nInt4PerChunk;
  }

  if (restNInt4 > 0) {
    for (size_t idx = 0; idx < restNInt4; idx += (WARP_SIZE / 8)) {
      size_t peerIdx = tidInWarp / 8;
      size_t int4Idx = tidInWarp % 8;
      int4 data;
      if (peerIdx != 7) {
        data = memChans[peerIdx].read<int4>(nInt4PerRank * rank + offsetOfThisBlock + idx + int4Idx);
      } else {
        data = buff4[nInt4PerRank * rank + offsetOfThisBlock + idx + int4Idx];
      }
      // Reduce to peerIdx 0
      for (int delta = (WARP_SIZE / 2); delta > 4; delta >>= 1) {
        int4 val;
        val.x = __shfl_down(data.x, delta);
        val.y = __shfl_down(data.y, delta);
        val.z = __shfl_down(data.z, delta);
        val.w = __shfl_down(data.w, delta);
        data = add_vectors<int>(val, data);
      }

      if (tidInWarp < (WARP_SIZE / 8)) {
        resultBuff4[offsetOfThisBlock + idx + int4Idx] = data;
      }
    }
 }

  __syncthreads();
  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    memChans[threadIdx.x].signal();
    memChans[threadIdx.x].wait();
  }
}

class ReduceScatterTestColl : public BaseTestColl {
 public:
  ReduceScatterTestColl() = default;
  ~ReduceScatterTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void ReduceScatterTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int nRanksPerNode = args.nRanksPerNode;
  const int kernelNum = args.kernelNum;
  int nBlocks;
  int nThreads;
  if (kernelNum == 5) {
    nBlocks = 8;
    nThreads = 256;
  } else if (kernelNum == 6) {
    nBlocks = 48;
    nThreads = 512;
  } else if (kernelNum == 7) {
    nBlocks = 56;
    nThreads = 512;
  } else if (kernelNum == 9) {
    nBlocks = 16;
    nThreads = 512;//(WARP_SIZE);
  } else {
    nBlocks = 1;
    nThreads = WARP_SIZE * (worldSize - 1);
  }
  if (nBlocks > MAX_BLOCKS) {
    std::stringstream err;
    err << "runColl invalid nBlocks " << nBlocks << " cannot be larger than " << MAX_BLOCKS;
    throw mscclpp::Error(err.str(), mscclpp::ErrorCode::InvalidUsage);
    return;
  }
  if (kernelNum == 5) {
    reducescatter5<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)outputBuff, rank, worldSize, nRanksPerNode,
                                                     paramCount_);
  } else if (kernelNum == 6) {
    reducescatter6<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)scratchBuff, (int*)outputBuff, rank, worldSize,
    nRanksPerNode, paramCount_);
  } else if (kernelNum == 7) {
    reducescatter7<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)scratchBuff, (int*)outputBuff, rank, worldSize,
    nRanksPerNode, paramCount_);
  } else if (kernelNum == 9) {
    reducescatter9<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)outputBuff, rank, worldSize, nRanksPerNode, paramCount_);
  }
}

void ReduceScatterTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<int> dataHost(sendCount_);
  for (int r = 0; r < worldSize; r++) {
    for (size_t i = 0; i < recvCount_; i++) {
      dataHost[r *recvCount_ + i] = rank + i;
    }
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = worldSize * (worldSize - 1) / 2 + (i * worldSize);
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void ReduceScatterTestColl::getBw(const double deltaSec, double& algBw, double& busBw) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;

  algBw = baseBw;
  double factor = ((double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void ReduceScatterTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / worldSize_);
  sendCount_ = base * worldSize_;
  recvCount_ = base;
  paramCount_ = sendCount_;
  expectedCount_ = recvCount_;
}

std::vector<KernelRestriction> ReduceScatterTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          // {0, "allgather0", true, 1, 4 * worldSize_},
          // {1, "allgather1", false, 1, 4 * worldSize_},
          // {2, "allgather2", true, 3, 4 * worldSize_},
          // {3, "allgather3", true, 1, 4 * worldSize_},
          // {4, "allgather4", true, 3, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {5, "reducescatter5", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {6, "reducescatter6", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {7, "reducescatter7", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {9, "reducescatter9", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/}};
}

class ReduceScatterTestEngine : public BaseTestEngine {
 public:
  ReduceScatterTestEngine(const TestArgs& args);
  ~ReduceScatterTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  bool isInPlace() const;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<char> scratchBuff_;
  std::shared_ptr<int> resultBuff_;
  std::shared_ptr<int[]> expectedBuff_;
  std::shared_ptr<mscclpp::LLPacket> scratchPacketBuff_;
  std::vector<mscclpp::MemoryChannel> smChannels_;
  std::vector<mscclpp::MemoryChannel> smOutOfPlaceChannels_;
};

ReduceScatterTestEngine::ReduceScatterTestEngine(const TestArgs& args) : BaseTestEngine(args, "allgather") {
  inPlace_ = isInPlace();
}

bool ReduceScatterTestEngine::isInPlace() const {
  // TODO
  return false;
}

void ReduceScatterTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  resultBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / NRANKS / sizeof(int)).memory();
  inputBuff = sendBuff_.get();
  outputBuff = resultBuff_.get();

  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / NRANKS / sizeof(int)]);

  if (args_.kernelNum == 6 || args_.kernelNum == 7) {
    scratchBuff_ = mscclpp::GpuBuffer<char>(SCRATCH_BUFF_SIZE).memory();
    scratchBuff = scratchBuff_.get();
  }
}

void ReduceScatterTestEngine::setupConnections() {
  setupMeshConnections(smChannels_, sendBuff_.get(), args_.maxBytes, nullptr, 0, ChannelSemantic::PUT, 64);
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> smChannelHandles(smChannels_.size());
  if (smChannels_.size() > sizeof(constMemChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
    std::runtime_error("unexpected error");
  }
  std::transform(smChannels_.begin(), smChannels_.end(), smChannelHandles.begin(),
                 [](const mscclpp::MemoryChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  CUDATHROW(cudaMemcpyToSymbol(constMemChans, smChannelHandles.data(),
                               sizeof(DeviceHandle<mscclpp::MemoryChannel>) * smChannelHandles.size()));

  if (args_.kernelNum == 6 || args_.kernelNum == 7) {
    setupMeshConnections(smOutOfPlaceChannels_, sendBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes,
                         ChannelSemantic::PUT, 64);
    std::vector<DeviceHandle<mscclpp::MemoryChannel>> smOutOfPlaceChannelHandles(smOutOfPlaceChannels_.size());
    if (smOutOfPlaceChannels_.size() > sizeof(constMemOutOfPlaceChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
      std::runtime_error("unexpected error");
    }
    std::transform(smOutOfPlaceChannels_.begin(), smOutOfPlaceChannels_.end(), smOutOfPlaceChannelHandles.begin(),
                   [](const mscclpp::MemoryChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    CUDATHROW(cudaMemcpyToSymbol(constMemOutOfPlaceChans, smOutOfPlaceChannelHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::MemoryChannel>) * smOutOfPlaceChannelHandles.size()));
  }
}

std::vector<void*> ReduceScatterTestEngine::getSendBuff() { return {sendBuff_.get()}; }

void* ReduceScatterTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* ReduceScatterTestEngine::getRecvBuff() {
  // in-place operation reuse the send buffer
  return resultBuff_.get();  // sendBuff_.get();
}

void* ReduceScatterTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<ReduceScatterTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<ReduceScatterTestColl>(); }
