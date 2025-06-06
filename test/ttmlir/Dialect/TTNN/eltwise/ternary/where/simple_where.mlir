// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module @jit_eltwise_where {
  func.func public @test_where(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = ttir.empty() : tensor<13x37xf32>
    %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    %2 = ttir.empty() : tensor<13x37xf32>
    %3 = "ttir.where"(%1, %arg0, %arg1, %2) : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    // CHECK: "ttnn.eq"
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: -> tensor<13x37xf32
    // CHECK: "ttnn.where"
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: -> tensor<13x37xf32
     return %3 : tensor<13x37xf32>
  }
}
