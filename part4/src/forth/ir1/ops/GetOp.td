//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

def GetOp : ForthOp<"get", [Pure]>
{
  let summary = "Helper operation";
  
  let description = [{
    Return the n's elements of the stack, beginning with 0 for the first element (TOS)
  }];
  
  let arguments = (ins  I32Tensor:$stack, UI32Attr:$n);

  let results = (outs I32);

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack, "unsigned int":$n)>
  ];
}
