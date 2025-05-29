// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "ttir-sparse-transform"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRSPARSETRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

/// Configuration for sparse transformation
struct SparseConfig {
  enum Mode { Auto, Structured2_4, Unstructured };
  Mode mode = Auto;
  float threshold = 0.5f;
  bool profiling = false;
  llvm::StringMap<float> sparsityRatios;
};

namespace profiling {
/// Dump a JSON file mapping each linalg.matmul op (by location) to its sparsity ratio.
void dumpSparsityStats(ModuleOp module) {
  std::error_code ec;
  llvm::raw_fd_ostream os("sparsity_stats.json", ec);
  if (ec) {
    llvm::errs() << "Error opening sparsity_stats.json: " << ec.message() << "\n";
    return;
  }
  llvm::json::OStream J(os);
  J.object([&]() {
    module.walk([&](linalg::MatmulOp op) {
      float ratio = 0.5f; // stub
      llvm::SmallString<64> buf;
      llvm::raw_svector_ostream locOs(buf);
      op.getLoc().print(locOs);
      llvm::StringRef locStr = buf.str();
      J.attribute(locStr, ratio);
    });
  });
  os << "\n";
  llvm::outs() << "Wrote sparsity stats to sparsity_stats.json\n";
}
} // namespace profiling

// Load JSON file of sparsity ratios into config.sparsityRatios
static void loadSparsityStats(llvm::StringRef filename, SparseConfig &config) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(filename);
  if (!bufOrErr)
    return;
  auto text = (*bufOrErr)->getBuffer();
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return;
  if (auto obj = parsed->getAsObject()) {
    for (auto &kv : *obj) {
      if (auto numOpt = kv.second.getAsNumber()) {
        // turn the JSON key into a string
        std::string key = kv.first.str();
        // store the ratio
        config.sparsityRatios[key] = *numOpt;
      }
    }
  }
}


/// Helper to create 2:4 structured mask metadata
static DictionaryAttr create2_4MaskMetadata(OpBuilder &builder,
                                            RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  SmallVector<int64_t> maskShape;
  for (auto dim : shape)
    maskShape.push_back((dim + 3) / 4);

  NamedAttribute maskFormat =
      builder.getNamedAttr("format", builder.getStringAttr("2:4_structured"));
  NamedAttribute maskShape_ =
      builder.getNamedAttr("mask_shape", builder.getI64ArrayAttr(maskShape));
  NamedAttribute originalShape =
      builder.getNamedAttr("original_shape", builder.getI64ArrayAttr(shape));
  NamedAttribute bitsPerGroup =
      builder.getNamedAttr("bits_per_group", builder.getI32IntegerAttr(2));
  NamedAttribute elementsPerGroup =
      builder.getNamedAttr("elements_per_group", builder.getI32IntegerAttr(4));

  return builder.getDictionaryAttr(
      {maskFormat, maskShape_, originalShape, bitsPerGroup, elementsPerGroup});
}

/// Helper to create unstructured mask metadata
static DictionaryAttr createUnstructuredMaskMetadata(OpBuilder &builder,
                                                     RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  SmallVector<int64_t> maskShape;
  for (auto dim : shape)
    maskShape.push_back((dim + 7) / 8);

  NamedAttribute maskFormat =
      builder.getNamedAttr("format", builder.getStringAttr("unstructured"));
  NamedAttribute maskShape_ =
      builder.getNamedAttr("mask_shape", builder.getI64ArrayAttr(maskShape));
  NamedAttribute originalShape =
      builder.getNamedAttr("original_shape", builder.getI64ArrayAttr(shape));
  NamedAttribute bitsPerElement =
      builder.getNamedAttr("bits_per_element", builder.getI32IntegerAttr(1));

  return builder.getDictionaryAttr({maskFormat, maskShape_, originalShape,
                                     bitsPerElement});
}

/// Pattern to transform dense matmul to sparse operations
class MatmulToSparsePattern : public OpRewritePattern<linalg::MatmulOp> {
  const SparseConfig &config;

public:
  MatmulToSparsePattern(MLIRContext *context, const SparseConfig &config)
      : OpRewritePattern<linalg::MatmulOp>(context), config(config) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Attempting to sparsify matmul: " << op << "\n");

    if (op->hasAttr("tt.sparse_mode"))
      return failure();
    if (!canSparsify(op))
      return failure();

    SparseConfig::Mode modeToUse = config.mode;
    if (config.mode == SparseConfig::Auto) {
      llvm::SmallString<64> buf;
      llvm::raw_svector_ostream locOs(buf);
      op.getLoc().print(locOs);
      llvm::StringRef locStr = buf.str();
      auto it = config.sparsityRatios.find(locStr);
      if (it == config.sparsityRatios.end() || it->second < config.threshold)
        return failure();
      modeToUse = (it->second <= 0.5f) ? SparseConfig::Structured2_4
                                      : SparseConfig::Unstructured;
    }
    StringRef sparseModeStr =
        (modeToUse == SparseConfig::Structured2_4 ? "2:4"
         : modeToUse == SparseConfig::Unstructured ? "unstructured"
                                                    : "auto");

    op->setAttr("tt.sparse_mode", rewriter.getStringAttr(sparseModeStr));
    op->setAttr("tt.sparse_threshold",
                rewriter.getF32FloatAttr(config.threshold));

    auto lhsType =
        cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
    auto rhsType =
        cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType());
    if (modeToUse == SparseConfig::Structured2_4) {
      op->setAttr("tt.sparse_lhs_mask",
                  create2_4MaskMetadata(rewriter, lhsType));
      op->setAttr("tt.sparse_rhs_mask",
                  create2_4MaskMetadata(rewriter, rhsType));
    } else if (modeToUse == SparseConfig::Unstructured) {
      op->setAttr("tt.sparse_lhs_mask",
                  createUnstructuredMaskMetadata(rewriter, lhsType));
      op->setAttr("tt.sparse_rhs_mask",
                  createUnstructuredMaskMetadata(rewriter, rhsType));
    }

    SmallVector<bool> sparseOperands = {true, true};
    op->setAttr("tt.sparse_operands",
                rewriter.getBoolArrayAttr(sparseOperands));

    LLVM_DEBUG(llvm::dbgs() << "Added sparse attributes to matmul: " << op
                            << "\n");
    return success();
  }

private:
  bool canSparsify(linalg::MatmulOp op) const {
    auto lhsType =
        cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
    auto rhsType =
        cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType());
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return false;
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];
    if (M < 64 && K < 64 && N < 64) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Matrix too small for sparsification (M=" << M
                 << ", K=" << K << ", N=" << N << ")\n");
      return false;
    }
    return true;
  }
};

/// Pass implementation
class TTIRSparseTransformPass
    : public impl::TTIRSparseTransformBase<TTIRSparseTransformPass> {
public:
  using impl::TTIRSparseTransformBase<
      TTIRSparseTransformPass>::TTIRSparseTransformBase;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running TTIR Sparse Transform Pass\n");

    SparseConfig config;
    if (mode == "structured-2-4")
      config.mode = SparseConfig::Structured2_4;
    else if (mode == "unstructured")
      config.mode = SparseConfig::Unstructured;
    config.threshold = sparsityThreshold;
    config.profiling = enableProfiling;

    if (config.profiling) {
      profiling::dumpSparsityStats(getOperation());
      return;
    }
    loadSparsityStats("sparsity_stats.json", config);

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToSparsePattern>(&getContext(), config);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::ttir
