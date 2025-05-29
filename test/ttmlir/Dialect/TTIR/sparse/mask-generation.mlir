// RUN: ttmlir-opt --ttir-sparse-transform="mode=structured-2-4" %s | FileCheck %s --check-prefix=CHECK-2-4
// RUN: ttmlir-opt --ttir-sparse-transform="mode=unstructured" %s | FileCheck %s --check-prefix=CHECK-UNSTRUCTURED

func.func @test_mask_generation(%arg0: tensor<128x64xf32>, 
                               %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x32xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  // CHECK-2-4: linalg.matmul
  // CHECK-2-4-SAME: tt.sparse_mode = "2:4"
  // CHECK-2-4-SAME: tt.sparse_lhs_mask = {
  // CHECK-2-4-SAME:   format = "2:4_structured"
  // CHECK-2-4-SAME:   mask_shape = [32 : i64, 16 : i64]
  // CHECK-2-4-SAME:   original_shape = [128 : i64, 64 : i64]
  // CHECK-2-4-SAME: }
  
  // CHECK-UNSTRUCTURED: linalg.matmul
  // CHECK-UNSTRUCTURED-SAME: tt.sparse_mode = "unstructured"
  // CHECK-UNSTRUCTURED-SAME: tt.sparse_lhs_mask = {
  // CHECK-UNSTRUCTURED-SAME:   format = "unstructured"
  // CHECK-UNSTRUCTURED-SAME:   mask_shape = [16 : i64, 8 : i64]
  // CHECK-UNSTRUCTURED-SAME: }
  
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) 
                          outs(%filled : tensor<128x32xf32>) -> tensor<128x32xf32>
  
  return %result : tensor<128x32xf32>
}
