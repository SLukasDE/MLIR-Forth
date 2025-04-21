//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

def PushOp : ForthOp<"push", [Pure]>
{
  let summary = "Helper operation";
  
  let description = [{
    It pushs an element from the stack
  }];
  
  let arguments = (ins  I32Tensor:$stack, I32:$value);

  let results = (outs I32Tensor);
  
  let assemblyFormat = [{
    `(` $stack `,` $value `:` type($stack) `,` type($value) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack, "mlir::Value":$value)>
  ];
}
