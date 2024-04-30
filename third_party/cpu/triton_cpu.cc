#include "TritonCPUToLLVM/Passes.h"
#include "TritonToTritonCPU/Passes.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_cpu_passes_ttc(py::module &&m) {
  m.def("add_triton_to_triton_cpu_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonToTritonCPUPipelineBuilder(pm);
  });
  m.def("add_triton_cpu_to_llvmir_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonCPUToLLVMPipelineBuilder(pm);
  });
}

void init_common_passes(py::module &m) {
  m.def("add_vector_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertVectorToLLVMPass());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_cpu_passes_ttc(passes.def_submodule("ttc"));
  init_common_passes(passes);

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect,
                    mlir::vector::VectorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("find_kernel_names", [](mlir::ModuleOp &mod) {
    std::vector<std::string> res;
    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      if (funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}
