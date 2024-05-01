#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

void PtrToMemRefOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {}

/// Parse an attribute registered to this dialect.
::mlir::Attribute
TritonCPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                 ::mlir::Type type) const {
  llvm_unreachable("parse stub called");
}

/// Print an attribute registered to this dialect.
void TritonCPUDialect::printAttribute(::mlir::Attribute attr,
                                      ::mlir::DialectAsmPrinter &os) const {
  llvm_unreachable("print stub called");
}

/// Parse a type registered to this dialect.
::mlir::Type
TritonCPUDialect::parseType(::mlir::DialectAsmParser &parser) const {
  llvm_unreachable("parse stub called");
}

/// Print a type registered to this dialect.
void TritonCPUDialect::printType(::mlir::Type type,
                                 ::mlir::DialectAsmPrinter &os) const {
  llvm_unreachable("print stub called");
}

void TritonCPUDialect::initialize() {
  // registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonCPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonCPU/IR/OpsEnums.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "triton/Dialect/TritonCPU/IR/Ops.cpp.inc"

// verify TritonCPU ops
LogicalResult TritonCPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
