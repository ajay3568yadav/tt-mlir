func.func @profile_test(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %filled = linalg.fill ins(%c0 : f32) 
                     outs(%init : tensor<128x128xf32>) 
                     -> tensor<128x128xf32>
  %res    = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) 
                         outs(%filled : tensor<128x128xf32>) 
                         -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
