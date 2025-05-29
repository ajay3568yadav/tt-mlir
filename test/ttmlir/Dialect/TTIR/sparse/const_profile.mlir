func.func @const_profile() -> tensor<128x128xf32> {
  // A full‐zero 128×128 tensor (sparsity = 1.0)
  %cst = arith.constant dense<0.0> : tensor<128x128xf32>

  // Prepare an empty output
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

  // This matmul is large enough (128 ≥ 64) and has 100% zeros
  %res = linalg.matmul
    ins(%cst, %cst : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%filled : tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
