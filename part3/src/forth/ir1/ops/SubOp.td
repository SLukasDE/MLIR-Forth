//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : ForthOp<"sub", [Pure]>
{
  let summary = "Forth - operation";
  
  let description = [{
    "Pops 2 elements from the stack, substract these elementes and pushs the result back to the stack"
  }];
  
  let arguments = (ins I32:$operand1, I32:$operand2);

  let results = (outs I32);
  
  let assemblyFormat = [{
    `(` $operand1 `,` $operand2 `:` type($operand1) `,` type($operand2) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$operand1, "mlir::Value":$operand2)>
  ];
}
