// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttir-lower-sparse-ops"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRLOWERSPARSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Pattern to lower sparse matmul to hardware operations
class LowerSparseMatmulPattern : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Only process operations with sparse attributes
    if (!op->hasAttr("tt.sparse_mode")) {
      return failure();
    }

    // Check if already lowered
    if (op->hasAttr("tt.sparse_lowered")) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Lowering sparse matmul: " << op << "\n");

    // Extract sparse attributes
    auto sparseModeAttr = op->getAttrOfType<StringAttr>("tt.sparse_mode");
    
    // Get the operands from the adaptor
    ValueRange operands = adaptor.getOperands();
    
    // Create a new linalg.matmul with the proper build signature
    // linalg.matmul expects inputs and outputs to be separated
    ValueRange inputs = operands.take_front(2);  // First two operands are inputs
    ValueRange outputs = operands.drop_front(2); // Remaining operands are outputs
    
    auto newOp = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), inputs, outputs);
    
    // Copy all existing attributes
    for (auto attr : op->getAttrs()) {
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    
    // Add lowering-specific attributes
    newOp->setAttr("tt.sparse_lowered", rewriter.getBoolAttr(true));
    
    if (sparseModeAttr.getValue() == "2:4") {
      newOp->setAttr("tt.hw_config", rewriter.getStringAttr("structured_sparse"));
      newOp->setAttr("tt.block_size", rewriter.getI32IntegerAttr(4));
      newOp->setAttr("tt.sparse_elements", rewriter.getI32IntegerAttr(2));
    } else if (sparseModeAttr.getValue() == "unstructured") {
      newOp->setAttr("tt.hw_config", rewriter.getStringAttr("unstructured_sparse"));
    } else {
      newOp->setAttr("tt.hw_config", rewriter.getStringAttr("auto_sparse"));
    }

    LLVM_DEBUG(llvm::dbgs() << "Created lowered sparse matmul: " << newOp << "\n");

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// Pass implementation
class TTIRLowerSparseOpsPass
    : public impl::TTIRLowerSparseOpsBase<TTIRLowerSparseOpsPass> {
public:
  using impl::TTIRLowerSparseOpsBase<TTIRLowerSparseOpsPass>::TTIRLowerSparseOpsBase;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running TTIR Lower Sparse Ops Pass\n");

    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Set up conversion target
    ConversionTarget target(*context);
    
    // Mark sparse matmuls as illegal (need to be converted)
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      // Legal if it doesn't have sparse attributes or if already lowered
      return !op->hasAttr("tt.sparse_mode") || op->hasAttr("tt.sparse_lowered");
    });
    
    // Everything else is legal
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Set up patterns
    RewritePatternSet patterns(context);
    patterns.add<LowerSparseMatmulPattern>(context);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Sparse lowering complete\n");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tt::TTDialect>();
    registry.insert<ttir::TTIRDialect>();
  }
};

} // namespace

} // namespace mlir::tt::ttir