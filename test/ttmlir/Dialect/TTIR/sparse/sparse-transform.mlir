// RUN: ttmlir-opt --ttir-sparse-transform %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_sparse_2_4
func.func @matmul_sparse_2_4(%arg0: tensor<128x64xf32>, 
                             %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  // CHECK: linalg.matmul
  // CHECK-SAME: tt.sparse_mode = "2:4"
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) 
                          outs(%filled : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  return %result : tensor<128x32xf32>
}

// CHECK-LABEL: func.func @small_matmul_not_sparsified
func.func @small_matmul_not_sparsified(%arg0: tensor<32x32xf32>, 
                                       %arg1: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32x16xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
  
  // CHECK: linalg.matmul
  // CHECK-NOT: tt.sparse_mode
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x16xf32>) 
                          outs(%filled : tensor<32x16xf32>) -> tensor<32x16xf32>
  
  return %result : tensor<32x16xf32>
}
