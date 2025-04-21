//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

def ExecuteOp : ForthOp<"execute", [Pure]>
{
  let summary = "Forth EXECUTE operation";
  
  let description = [{
    "Pops the TOS element and runs the word found at that address"
  }];
  
  let arguments = (ins I32Tensor:$stack, I32:$addr);

  let results = (outs I32Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack, "mlir::Value":$addr)>
  ];
}
