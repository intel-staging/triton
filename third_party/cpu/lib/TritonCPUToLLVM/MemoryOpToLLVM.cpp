#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetPlatform.hpp"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMORYOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

// TODO: use enums to access struct fields.
struct PtrToMemRefOpConversion : public OpConversionPattern<PtrToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PtrToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tensorPtrStruct = rewriter.getRemappedValue(op.getSrc());
    auto memRefTy = op.getType().cast<MemRefType>();
    auto rank = memRefTy.getRank();
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());
    auto memRefStructFields =
        memRefStructTy.cast<LLVM::LLVMStructType>().getBody();
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto copyValue = [&](Value to, int64_t idxFrom, int64_t idxTo) {
      auto valueTy = memRefStructFields[idxTo];
      Value val = rewriter.create<LLVM::ExtractValueOp>(
          loc, valueTy, tensorPtrStruct, idxFrom);
      return rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, to, val,
                                                  idxTo);
    };

    Value res = undef(memRefStructTy);
    // Copy base.
    res = copyValue(res, 0, 1);
    // Compute offset and add it to the base.
    Value offset;
    for (int64_t i = 0; i < rank; i++) {
      Value dimOffs = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, tensorPtrStruct, SmallVector<int64_t, 2>{1, i});
      Value dimStride = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, tensorPtrStruct, SmallVector<int64_t, 2>{3, i});
      Value offsInElems = rewriter.create<LLVM::MulOp>(loc, dimOffs, dimStride);
      offset = offset ? rewriter.create<LLVM::AddOp>(loc, offset, offsInElems)
                      : offsInElems;
    }
    Value base = rewriter.create<LLVM::ExtractValueOp>(
        loc, memRefStructFields[1], tensorPtrStruct, 0);
    base = rewriter.create<LLVM::GEPOp>(
        loc, base.getType(), memRefTy.getElementType(), base, offset);
    res =
        rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res, base, 1);
    // Use 0 offset.
    res = rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res,
                                               i64_val(0), 2);
    // Copy shape.
    res = copyValue(res, 2, 3);
    // Copy strides.
    res = copyValue(res, 3, 4);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct MakeTensorPtrOpConversion : public OpConversionPattern<MakeTensorPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structTy = getTypeConverter()->convertType(op.getType());
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto insertArray = [&](Value structVal, auto values, int64_t idx,
                           Type zextTo = nullptr) {
      for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        Value val = values[i];
        if (zextTo)
          val = rewriter.create<LLVM::ZExtOp>(loc, zextTo, val);
        structVal = rewriter.create<LLVM::InsertValueOp>(
            loc, structTy, structVal, val, SmallVector<int64_t, 2>{idx, i});
      }
      return structVal;
    };

    Value res = undef(structTy);
    // 0 - base pointer.
    auto base = rewriter.getRemappedValue(op.getBase());
    res = rewriter.create<LLVM::InsertValueOp>(loc, structTy, res, base, 0);
    // 1 - array<rank> for offsets. Promote values to i64.
    res = insertArray(res, op.getOffsets(), 1, i64Ty);
    // 2 - array<rank> for shape.
    res = insertArray(res, op.getShape(), 2);
    // 3 - array<rank> for strides.
    res = insertArray(res, op.getStrides(), 3);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct AdvanceOpConversion : public OpConversionPattern<AdvanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = IntegerType::get(getContext(), 64);
    Value res = rewriter.getRemappedValue(op.getPtr());
    Type structTy = res.getType();
    auto offsets = op.getOffsets();

    for (int64_t i = 0; i < offsets.size(); ++i) {
      auto oldOffset = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, res, SmallVector<int64_t, 2>{1, i});
      auto step = rewriter.create<LLVM::SExtOp>(loc, i64Ty, offsets[i]);
      auto newOffset = rewriter.create<LLVM::AddOp>(loc, oldOffset, step);
      res = rewriter.create<LLVM::InsertValueOp>(loc, structTy, res, newOffset,
                                                 SmallVector<int64_t, 2>{1, i});
    }

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct MemoryOpToLLVM
    : public triton::impl::MemoryOpToLLVMBase<MemoryOpToLLVM> {
  using MemoryOpToLLVMBase::MemoryOpToLLVMBase;

  MemoryOpToLLVM() : MemoryOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<PtrToMemRefOpConversion>(typeConverter, context);
    patterns.add<MakeTensorPtrOpConversion>(typeConverter, context);
    patterns.add<AdvanceOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass() {
  return std::make_unique<MemoryOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
