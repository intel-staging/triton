#include "TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

TritonCPUToLLVMTypeConverter::TritonCPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
}

Type TritonCPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (pointeeType.isa<RankedTensorType>()) {
    // struct {
    //   ptr base_ptr;
    //   array<rank x i64> offsets;
    //   array<rank x i64> shape;
    //   array<rank x i64> strides;
    // }
    auto tensorTy = pointeeType.cast<RankedTensorType>();
    auto rank = tensorTy.getShape().size();
    auto i64Ty = IntegerType::get(ctx, 64);
    SmallVector<Type, 4> types;
    types.push_back(LLVM::LLVMPointerType::get(ctx));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx);
}
