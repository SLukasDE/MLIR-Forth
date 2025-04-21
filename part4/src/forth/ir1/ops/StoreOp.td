//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : ForthOp<"store", [Pure]>
{
  let summary = "Forth ! operation";
  
  let description = [{
    "Stores a value from the stack at an address from the stack"
  }];
  
  let arguments = (ins  I32Tensor:$stack);

  let results = (outs I32Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack)>
  ];
}
