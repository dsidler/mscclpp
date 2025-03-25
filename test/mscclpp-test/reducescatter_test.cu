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

// Assume out-of-place
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
  constexpr size_t nInt4PerChunk = 1024 * 256 / sizeof(int4);  // 256KB
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
    nBlocks = 35;
    nThreads = 512;
  } else if (kernelNum == 7) {
    nBlocks = 8;
    nThreads = 256;  // 512 works not too bad for small sizes;
  } else {
    nBlocks = 1;
    nThreads = WARP_SIZE * (worldSize - 1);
  }
  if (kernelNum == 5) {
    reducescatter5<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)outputBuff, rank, worldSize, nRanksPerNode,
                                                     paramCount_);
  } else if (kernelNum == 6) {
    // reducescatter6<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)scratchBuff, rank, worldSize,
    // nRanksPerNode, paramCount_);
  } else if (kernelNum == 7) {
    // reducescatter7<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, (int*)outputBuff, rank, worldSize,
    // nRanksPerNode, paramCount_);
  }
}

void ReduceScatterTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = worldSize * (worldSize - 1) / 2;
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
          {5, "reducescatter5", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/}};
  // {6, "reducescatter6", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
  // {7, "reducescatter7", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/}};
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
  std::shared_ptr<int> scratchBuff_;
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

  if (args_.kernelNum == 6) {
    scratchBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
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

  if (args_.kernelNum == 6) {
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
