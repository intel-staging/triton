#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#define GEN_PASS_DEF_FUNCOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendProgramIdArgs(triton::FuncOp funcOp,
                                    ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    // 1. Modify the function type to add new arguments.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(i32_ty);
    amendedInputTy.push_back(i32_ty);
    amendedInputTy.push_back(i32_ty);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add new arguments.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new arguments to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(funcTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto modifiedFuncOp = funcOp;
    if (LLVM::isKernel(funcOp))
      modifiedFuncOp = amendProgramIdArgs(modifiedFuncOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        modifiedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp)
      return failure();

    // required by AxisInfoAnalysis
    if (LLVM::isKernel(funcOp))
      rewriter.eraseOp(modifiedFuncOp);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct FuncOpToLLVM : public triton::impl::FuncOpToLLVMBase<FuncOpToLLVM> {
  using FuncOpToLLVMBase::FuncOpToLLVMBase;

  FuncOpToLLVM() : FuncOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower tt.func
    RewritePatternSet funcPatterns(context);
    funcPatterns.add<FuncOpConversion>(typeConverter,
                                       /*benefit=*/1);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          funcPatterns);
    if (failed(
            applyPartialConversion(mod, convTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // Lower tt.call, tt.return
    // TODO: use CPU-specific lowering patterns.
    int benefit = 10;
    RewritePatternSet patterns(context);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     benefit);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createFuncOpToLLVMPass() {
  return std::make_unique<FuncOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
