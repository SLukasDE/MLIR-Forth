//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : ForthOp<"load", [Pure]>
{
  let summary = "Forth @ operation";
  
  let description = [{
    "Loads the value from the given address and puts that value on the stack"
  }];
  
  let arguments = (ins  I32Tensor:$stack);

  let results = (outs I32Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack)>
  ];
}
