#ifndef TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonCPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonCPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace cpu {} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_
