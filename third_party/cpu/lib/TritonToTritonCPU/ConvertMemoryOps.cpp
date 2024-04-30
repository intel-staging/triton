#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTMEMORYOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonPointerConverter : public TypeConverter {
public:
  TritonPointerConverter() {
    // Convert pointers to unranked memrefs.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrTy) -> Type {
      if (triton::isTensorPointerType(ptrTy)) {
        auto tensorTy = ptrTy.getPointeeType().dyn_cast<RankedTensorType>();
        auto elemTy = tensorTy.getElementType();
        auto shape = tensorTy.getShape();
        return MemRefType::get(shape, elemTy);
        // return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
      }
      return ptrTy;
    });
    addConversion([](RankedTensorType tensorTy) -> Type {
      return VectorType::get(tensorTy.getShape(), tensorTy.getElementType());
    });

    // Converted loads produce vectors instead of tensors. Provide conversion
    // here for users.
    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    });

    // Converted loads and stores consume memrefs instead of pointers and
    // vectors instead of tensor. Provide conversion here for uses.
    addTargetMaterialization(
        [&](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          if (type.isa<VectorType>()) {
            auto cast =
                builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
            return std::optional<Value>(cast.getResult(0));
          } else if (type.isa<MemRefType>()) {
            auto cast = builder.create<PtrToMemRefOp>(loc, type, inputs);
            return std::optional<Value>(cast.getResult());
          }
          llvm_unreachable("Unexpected target materizalization");
        });
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto mask = loadOp.getMask();
    auto ptr = loadOp.getPtr();
    auto boundaryChecks = loadOp.getBoundaryCheck();

    // TODO: support masks, tensors of pointers, boundary checks.
    if (mask || !triton::isTensorPointerType(ptr.getType()) ||
        !boundaryChecks.empty()) {
      llvm_unreachable("unsupported load op");
    }

    // Replace tt.load with vector.load.
    auto memRef = rewriter.getRemappedValue(ptr);
    auto rank = memRef.getType().dyn_cast<MemRefType>().getRank();
    auto resTy = getTypeConverter()->convertType(loadOp.getResult().getType());
    Value zero_cst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(rank, zero_cst);
    auto vecLoad = rewriter.create<vector::LoadOp>(loc, resTy, memRef, indices);
    rewriter.replaceOp(loadOp, vecLoad);

    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto mask = storeOp.getMask();
    auto ptr = storeOp.getPtr();
    auto boundaryChecks = storeOp.getBoundaryCheck();

    // TODO: support masks, tensors of pointers, boundary checks.
    if (mask || !triton::isTensorPointerType(ptr.getType()) ||
        !boundaryChecks.empty()) {
      llvm_unreachable("unsupported store op");
    }

    // Replace tt.store with vector.store.
    auto value = rewriter.getRemappedValue(storeOp.getValue());
    auto memRef = rewriter.getRemappedValue(ptr);
    auto rank = memRef.getType().dyn_cast<MemRefType>().getRank();
    Value zero_cst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(rank, zero_cst);
    auto vecStore =
        rewriter.create<vector::StoreOp>(loc, value, memRef, indices);
    rewriter.replaceOp(storeOp, vecStore);

    return success();
  }
};

class MemoryOpConversionTarget : public ConversionTarget {
public:
  explicit MemoryOpConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addIllegalOp<triton::LoadOp>();
    addIllegalOp<triton::StoreOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertMemoryOps
    : public triton::impl::ConvertMemoryOpsBase<ConvertMemoryOps> {
  using ConvertMemoryOpsBase::ConvertMemoryOpsBase;

  ConvertMemoryOps() : ConvertMemoryOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    MemoryOpConversionTarget convTarget(*context);
    TritonPointerConverter pointerConverter;
    RewritePatternSet patterns(context);
    patterns.add<LoadOpConversion>(pointerConverter, context);
    patterns.add<StoreOpConversion>(pointerConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps() {
  return std::make_unique<ConvertMemoryOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
