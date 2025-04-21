//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

def PopOp : ForthOp<"pop", [Pure]>
{
  let summary = "Helper operation";
  
  let description = [{
    It pops n elements from the stack
  }];
  
  let arguments = (ins  I32Tensor:$stack, UI32Attr:$n);

  let results = (outs I32Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack, "unsigned int":$n)>
  ];
}
