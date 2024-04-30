#ifndef TRITONCPU_ATTRDEFS
#define TRITONCPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonCPU Attribute Definitions
//===----------------------------------------------------------------------===//
class TritonCPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonCPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, traits, baseCppClass> {

  let description = [{TritonCPU attr.}];
  let attrName = "triton.cpu." # attrMnemonic;
}

#endif
