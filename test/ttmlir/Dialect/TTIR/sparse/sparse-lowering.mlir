// RUN: ttmlir-opt --ttir-sparse-transform --ttir-lower-sparse-ops %s | FileCheck %s

// CHECK-LABEL: func.func @test_sparse_lowering
func.func @test_sparse_lowering(%arg0: tensor<128x64xf32>, 
                               %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  // After sparse transform and lowering:
  // CHECK-NOT: linalg.matmul
  // CHECK: ttir.sparse_matmul
  // CHECK-SAME: sparse_mode = "auto"
  // CHECK-SAME: threshold = 5.000000e-01 : f32
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) 
                          outs(%filled : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  return %result : tensor<128x32xf32>
}

// CHECK-LABEL: func.func @test_2_4_lowering
func.func @test_2_4_lowering(%arg0: tensor<256x128xf32>, 
                            %arg1: tensor<128x64xf32>) -> tensor<256x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<256x64xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<256x64xf32>) -> tensor<256x64xf32>
  
  // With 2:4 mode:
  // CHECK: ttir.sparse_matmul
  // CHECK-SAME: sparse_mode = "2:4"
  // CHECK-SAME: hw_config = "structured_sparse"
  // CHECK-SAME: block_size = 4 : i32
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<256x128xf32>, tensor<128x64xf32>) 
                          outs(%filled : tensor<256x64xf32>) -> tensor<256x64xf32>
  
  return %result : tensor<256x64xf32>
}