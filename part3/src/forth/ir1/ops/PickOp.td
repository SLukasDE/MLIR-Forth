//===----------------------------------------------------------------------===//
// PickOp
//===----------------------------------------------------------------------===//

def PickOp : ForthOp<"pick", [Pure]>
{
  let summary = "Helper operation, it's not the original Forth PICK operation";
  
  let description = [{
    This op is equal to GetOp, but it takes an dynamic value/type as 2nd argument, instead of an attribute. 
  }];
  
  let arguments = (ins I32Tensor:$stack, I32:$n);

  let results = (outs I32);
  
  let assemblyFormat = [{
    `(` $stack `[` $n `]` `:` type($stack) `,` type($n) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$stack, "mlir::Value":$n)>
  ];
}
