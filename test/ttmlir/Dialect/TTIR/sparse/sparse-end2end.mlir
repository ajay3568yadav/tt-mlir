// RUN: ttmlir-opt --ttir-sparse-transform --ttir-lower-sparse-ops %s | FileCheck %s

func.func @test_end2end(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %c0     = arith.constant 0.0 : f32
  %init   = tensor.empty() : tensor<128x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  %res    = linalg.matmul 
             ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) 
             outs(%filled : tensor<128x32xf32>) 
             -> tensor<128x32xf32>
  return %res : tensor<128x32xf32>
}

// CHECK: tt.sparse_lowered = true
// CHECK: tt.hw_config = "structured_sparse"
