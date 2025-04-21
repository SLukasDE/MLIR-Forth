//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : ForthOp<"add", [Pure]>
{
  let summary = "Forth + operation";
  
  let description = [{
    Adds 2 elements and return the result
  }];
  
  let arguments = (ins I32:$operand1, I32:$operand2);

  let results = (outs I32);

  let builders = [
    OpBuilder<(ins "mlir::Value":$operand1, "mlir::Value":$operand2)>
  ];
  
  let assemblyFormat = [{
    `(` $operand1 `,` $operand2 `:` type($operand1) `,` type($operand2) `)` attr-dict `to` type(results)
  }];

  let extraClassDeclaration = [{
    static int getNumberOfOperands(){
      return 2;
    }
    static int getNumberOfResults(){
      return 1;
    }
  }];
}
