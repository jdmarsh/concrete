// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "concretelang/Support/Constants.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

namespace TFHE = mlir::concretelang::TFHE;

namespace {
struct TFHEGlobalParametrizationPass
    : public TFHEGlobalParametrizationBase<TFHEGlobalParametrizationPass> {
  TFHEGlobalParametrizationPass(
      mlir::concretelang::V0Parameter &cryptoParameters)
      : cryptoParameters(cryptoParameters){};
  void runOnOperation() final;
  mlir::concretelang::V0Parameter &cryptoParameters;
};
} // namespace

using mlir::concretelang::TFHE::GLWECipherTextType;

/// TFHEGlobalParametrizationTypeConverter is a TypeConverter that transform
/// `TFHE.glwe<sk[?]>` to
/// `TFHE.glwe<sk[id]<glweDimension,polynomialSize>>`
class TFHEGlobalParametrizationTypeConverter : public mlir::TypeConverter {

public:
  TFHEGlobalParametrizationTypeConverter(
      mlir::concretelang::V0Parameter &cryptoParameters)
      : cryptoParameters(cryptoParameters) {
    addConversion([](mlir::Type type) { return type; });
    addConversion(
        [&](GLWECipherTextType type) { return this->glweInterPBSType(type); });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(type.getShape(),
                                                 this->glweInterPBSType(glwe));
      return r;
    });
    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::FutureType>()
                                .getElementType()));
    });
    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::PointerType>()
                                .getElementType()));
    });
  }

  TFHE::GLWESecretKey getInterPBSKey() {
    auto dimension = cryptoParameters.getNBigLweDimension();
    auto polynomialSize = 1;
    // Warning, for now we use hardcoded ids. Later on, we expect the optimizer
    // to give the id.
    auto id = 1;
    return mlir::concretelang::TFHE::GLWESecretKey(dimension, polynomialSize,
                                                   id);
  }

  TFHE::GLWECipherTextType glweInterPBSType(GLWECipherTextType &type) {
    return TFHE::GLWECipherTextType::get(type.getContext(), getInterPBSKey());
  }

  TFHE::GLWESecretKey getIntraPBSKey() {
    auto dimension = cryptoParameters.nSmall;
    auto polynomialSize = 1;
    // Warning, for now we use hardcoded ids. Later on, we expect the optimizer
    // to give the id.
    auto id = 3;
    return mlir::concretelang::TFHE::GLWESecretKey(dimension, polynomialSize,
                                                   id);
  }

  TFHE::GLWECipherTextType glweIntraPBSType(GLWECipherTextType &type) {
    return TFHE::GLWECipherTextType::get(type.getContext(), getIntraPBSKey());
  }

  mlir::concretelang::V0Parameter cryptoParameters;
};

struct KeySwitchGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp> {
  KeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                         TFHEGlobalParametrizationTypeConverter &converter,
                         mlir::concretelang::V0Parameter &cryptoParameters,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp>(context, benefit),
        converter(converter), cryptoParameters(cryptoParameters) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::KeySwitchGLWEOp ksOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        ksOp.getCiphertext().getType().cast<TFHE::GLWECipherTextType>();
    auto newInputTy = converter.convertType(inputTy)
                          .cast<mlir::concretelang::TFHE::GLWECipherTextType>();
    auto outputTy = ksOp.getResult().getType().cast<TFHE::GLWECipherTextType>();
    auto newOutputTy = converter.glweIntraPBSType(outputTy);
    auto newInputKey = converter.getInterPBSKey();
    auto newOutputKey = converter.getIntraPBSKey();
    auto keyswitchKey = TFHE::GLWEKeyswitchKeyAttr::get(
        ksOp->getContext(), newInputKey, newOutputKey, cryptoParameters.ksLevel,
        cryptoParameters.ksLogBase);
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::KeySwitchGLWEOp>(
        ksOp, newOutputTy, ksOp.getCiphertext(), keyswitchKey);
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertext().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  TFHEGlobalParametrizationTypeConverter &converter;
  mlir::concretelang::V0Parameter &cryptoParameters;
};

struct BootstrapGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::BootstrapGLWEOp> {
  BootstrapGLWEOpPattern(mlir::MLIRContext *context,
                         TFHEGlobalParametrizationTypeConverter &converter,
                         mlir::concretelang::V0Parameter &cryptoParameters,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::BootstrapGLWEOp>(context, benefit),
        converter(converter), cryptoParameters(cryptoParameters) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::BootstrapGLWEOp bsOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        bsOp.getCiphertext().getType().cast<TFHE::GLWECipherTextType>();
    auto newInputTy = converter.glweIntraPBSType(inputTy);
    auto outputTy = bsOp.getResult().getType().cast<TFHE::GLWECipherTextType>();
    auto newOutputTy =
        converter.convertType(outputTy).cast<TFHE::GLWECipherTextType>();
    auto newInputKey = converter.getIntraPBSKey();
    auto newOutputKey = converter.getInterPBSKey();
    auto bootstrapKey = TFHE::GLWEBootstrapKeyAttr::get(
        bsOp->getContext(), newInputKey, newOutputKey,
        cryptoParameters.getPolynomialSize(), cryptoParameters.glweDimension,
        cryptoParameters.brLevel, cryptoParameters.brLogBase);
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::BootstrapGLWEOp>(
        bsOp, newOutputTy, bsOp.getCiphertext(), bsOp.getLookupTable(),
        bootstrapKey);
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertext().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  TFHEGlobalParametrizationTypeConverter &converter;
  mlir::concretelang::V0Parameter &cryptoParameters;
};

struct WopPBSGLWEOpPattern : public mlir::OpRewritePattern<TFHE::WopPBSGLWEOp> {
  WopPBSGLWEOpPattern(mlir::MLIRContext *context,
                      TFHEGlobalParametrizationTypeConverter &converter,
                      mlir::concretelang::V0Parameter &cryptoParameters,
                      mlir::PatternBenefit benefit =
                          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::WopPBSGLWEOp>(context, benefit),
        converter(converter), cryptoParameters(cryptoParameters) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::WopPBSGLWEOp wopPBSOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        wopPBSOp.getCiphertexts().getType().cast<mlir::RankedTensorType>();
    auto newInputTy =
        converter.convertType(inputTy).cast<mlir::RankedTensorType>();
    auto outputTy = wopPBSOp.getType().cast<mlir::RankedTensorType>();
    auto newOutputType = converter.convertType(outputTy);
    auto interKey = converter.getInterPBSKey();
    auto intraKey = converter.getIntraPBSKey();
    auto keyswitchKey = TFHE::GLWEKeyswitchKeyAttr::get(
        wopPBSOp->getContext(), interKey, intraKey, cryptoParameters.ksLevel,
        cryptoParameters.ksLogBase);
    auto bootstrapKey = TFHE::GLWEBootstrapKeyAttr::get(
        wopPBSOp->getContext(), intraKey, interKey,
        cryptoParameters.getPolynomialSize(), cryptoParameters.glweDimension,
        cryptoParameters.brLevel, cryptoParameters.brLogBase);
    auto packingKeyswitchKey = TFHE::GLWEPackingKeyswitchKeyAttr::get(
        wopPBSOp->getContext(), interKey, interKey,
        cryptoParameters.largeInteger->wopPBS.packingKeySwitch
            .outputPolynomialSize,
        cryptoParameters.largeInteger->wopPBS.packingKeySwitch
            .inputLweDimension,
        cryptoParameters.largeInteger->wopPBS.packingKeySwitch.level,
        cryptoParameters.largeInteger->wopPBS.packingKeySwitch.baseLog);
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::WopPBSGLWEOp>(
        wopPBSOp, newOutputType, wopPBSOp.getCiphertexts(),
        wopPBSOp.getLookupTable(), keyswitchKey, bootstrapKey,
        packingKeyswitchKey,
        rewriter.getI64ArrayAttr(
            cryptoParameters.largeInteger->crtDecomposition),
        rewriter.getI32IntegerAttr(
            cryptoParameters.largeInteger->wopPBS.circuitBootstrap.level),
        rewriter.getI32IntegerAttr(
            cryptoParameters.largeInteger->wopPBS.circuitBootstrap.baseLog));
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertexts().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  TFHEGlobalParametrizationTypeConverter &converter;
  mlir::concretelang::V0Parameter &cryptoParameters;
};

template <typename Op>
void populateWithTFHEOpTypeConversionPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<Op>>(
      patterns.getContext(), typeConverter);

  target.addDynamicallyLegalOp<Op>(
      [&](Op op) { return typeConverter.isLegal(op->getResultTypes()); });
}

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithTFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::ZeroGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::ZeroTensorGLWEOp>(patterns, target,
                                                  typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::AddGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::SubGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::NegGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::MulGLWEIntOp>(patterns, target, typeConverter);
}

void TFHEGlobalParametrizationPass::runOnOperation() {
  auto op = this->getOperation();

  TFHEGlobalParametrizationTypeConverter converter(cryptoParameters);

  // Parametrize
  {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // function signature
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
        [&](mlir::func::ConstantOp op) {
          return FunctionConstantOpConversion<
              TFHEGlobalParametrizationTypeConverter>::isLegal(op, converter);
        });
    patterns.add<
        FunctionConstantOpConversion<TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);

    // Parametrize keyswitch bootstrap
    target.addLegalOp<mlir::arith::ConstantOp>();
    patterns.add<KeySwitchGLWEOpPattern>(&getContext(), converter,
                                         cryptoParameters);
    target.addDynamicallyLegalOp<TFHE::KeySwitchGLWEOp>(
        [&](TFHE::KeySwitchGLWEOp op) {
          return !op.getKey().getInputKey().isNotParameterized() &&
                 !op.getKey().getOutputKey().isNotParameterized() &&
                 op.getKey().getBaseLog() != 0 && op.getKey().getLevels() != 0;
        });
    patterns.add<BootstrapGLWEOpPattern>(&getContext(), converter,
                                         cryptoParameters);
    target.addDynamicallyLegalOp<TFHE::BootstrapGLWEOp>(
        [&](TFHE::BootstrapGLWEOp op) {
          return converter.isLegal(op->getResultTypes());
        });

    // Parametrize wop pbs
    patterns.add<WopPBSGLWEOpPattern>(&getContext(), converter,
                                      cryptoParameters);
    target.addDynamicallyLegalOp<TFHE::WopPBSGLWEOp>(
        [&](TFHE::WopPBSGLWEOp op) {
          return !op.getType()
                      .cast<mlir::RankedTensorType>()
                      .getElementType()
                      .cast<TFHE::GLWECipherTextType>()
                      .hasUnparametrizedParameters();
        });

    // Add all patterns to convert TFHE types
    populateWithTFHEOpTypeConversionPatterns(patterns, target, converter);

    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::bufferization::AllocTensorOp>>(&getContext(), converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::bufferization::AllocTensorOp>(target, converter);

    patterns.add<RegionOpTypeConverterPattern<
        mlir::linalg::GenericOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::tensor::GenerateOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::scf::ForOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::func::ReturnOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
        target, converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::linalg::YieldOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::linalg::YieldOp>(
        target, converter);

    mlir::concretelang::populateWithTensorTypeConverterPatterns(
        patterns, target, converter);

    // Conversion of RT Dialect Ops
    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::Tracing::TraceCiphertextOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::scf::YieldOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::MakeReadyFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::AwaitFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::CreateAsyncTaskOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::WorkFunctionReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::RegisterTaskWorkFunctionOp>>(&getContext(),
                                                                 converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::Tracing::TraceCiphertextOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::MakeReadyFutureOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::AwaitFutureOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::CreateAsyncTaskOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::WorkFunctionReturnOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::RegisterTaskWorkFunctionOp>(target, converter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFHEGlobalParametrizationPass(
    mlir::concretelang::V0FHEContext &fheContext) {
  return std::make_unique<TFHEGlobalParametrizationPass>(fheContext.parameter);
}
} // namespace concretelang
} // namespace mlir
