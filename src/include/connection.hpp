#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include "communicator.hpp"
#include "ib.hpp"
#include "mscclpp.hpp"
#include <cuda_runtime.h>

namespace mscclpp {

// TODO: Add functionality to these classes for Communicator to do connectionSetup

class ConnectionBase : public Connection
{
  int remoteRank_;
  int tag_;
public:
  ConnectionBase(int remoteRank, int tag);

  int remoteRank() override;
  int tag() override;

  virtual void startSetup(std::shared_ptr<BaseBootstrap>){};
  virtual void endSetup(std::shared_ptr<BaseBootstrap>){};
};

class CudaIpcConnection : public ConnectionBase
{
  cudaStream_t stream;

public:
  CudaIpcConnection(int remoteRank, int tag);

  ~CudaIpcConnection();

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;

  void flush() override;
};

class IBConnection : public ConnectionBase
{
  Transport transport_;
  Transport remoteTransport_;
  IbQp* qp;

public:
  IBConnection(int remoteRank, int tag, Transport transport, Communicator::Impl& commImpl);

  ~IBConnection();

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;

  void flush() override;

  void startSetup(std::shared_ptr<BaseBootstrap> bootstrap) override;

  void endSetup(std::shared_ptr<BaseBootstrap> bootstrap) override;
};

} // namespace mscclpp

#endif // MSCCLPP_CONNECTION_HPP_
