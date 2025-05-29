// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace mlir;

namespace tt {
namespace ttm {
namespace ttir {

static std::optional<float> computeZeroDensity(Value v) {
  // Only handle constant tensor inputs
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute val = constOp.getValue();
    // Try casting to a dense elements attribute
    if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(val)) {
      uint64_t total = 0, zeros = 0;
      for (Attribute eltAttr : denseAttr.getValues<Attribute>()) {
        if (auto f = llvm::dyn_cast<FloatAttr>(eltAttr)) {
          ++total;
          if (f.getValue().isZero())
            ++zeros;
        } else if (auto i = llvm::dyn_cast<IntegerAttr>(eltAttr)) {
          ++total;
          if (i.getValue() == 0)
            ++zeros;
        }
      }
      if (total == 0)
        return std::nullopt;
      return float(zeros) / float(total);
    }
  }
  return std::nullopt;
}

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
      Value lhs = op.getDpsInputOperand(0)->get();
      Value rhs = op.getDpsInputOperand(1)->get();

      auto lhsDen = computeZeroDensity(lhs);
      auto rhsDen = computeZeroDensity(rhs);
      if (!lhsDen.has_value() || !rhsDen.has_value())
        return;  // skip non-constant cases

      float ratio = (lhsDen.value() + rhsDen.value()) * 0.5f;

      // Render location to string
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

} // namespace ttir
} // namespace ttm
} // namespace tt
