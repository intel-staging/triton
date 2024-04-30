#ifndef TRITONTOTRITONCPU_CONVERSION_PASSES
#define TRITONTOTRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemoryOps : Pass<"triton-cpu-convert-memory-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertMemoryOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertElementwiseOps : Pass<"triton-cpu-convert-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton elementwise ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertElementwiseOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
