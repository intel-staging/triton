#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
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
#define GEN_PASS_DEF_CONVERTELEMENTWISEOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonElementwiseConverter : public TypeConverter {
public:
  TritonElementwiseConverter() {
    // Convert tensors to vectors.
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType tensorTy) -> Type {
      return VectorType::get(tensorTy.getShape(), tensorTy.getElementType());
    });

    // Provide conversion between tensors and vectors for users and uses.
    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    });
    addTargetMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    });
  }
};

class ElementwiseOpConversionTarget : public ConversionTarget {
public:
  explicit ElementwiseOpConversionTarget(MLIRContext &ctx,
                                         TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalOp<arith::AddFOp>(
        [&](Operation *op) -> std::optional<bool> {
          return converter.isLegal(op);
        });
  }
};

template <typename OpT>
struct ElementwiseOpConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getTypeConverter;
  using typename OpConversionPattern<OpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState newState(op.getLoc(), op->getName());
    // Convert operands.
    for (auto operand : op->getOperands()) {
      Value newOperand = rewriter.getRemappedValue(operand);
      newState.operands.push_back(newOperand);
    }
    // Convert result types.
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newState.types))) {
      return failure();
    }
    newState.attributes = op->getAttrs();

    auto newOp = rewriter.create(newState);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ConvertElementwiseOps
    : public triton::impl::ConvertElementwiseOpsBase<ConvertElementwiseOps> {
  using ConvertElementwiseOpsBase::ConvertElementwiseOpsBase;

  ConvertElementwiseOps() : ConvertElementwiseOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonElementwiseConverter typeConverter;
    ElementwiseOpConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ElementwiseOpConversion<arith::AddFOp>>(typeConverter,
                                                         context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertElementwiseOps() {
  return std::make_unique<ConvertElementwiseOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
