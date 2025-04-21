//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : ForthOp<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "call operation of an forth-word";
  let description = [{
    Represent calls to a user defined word. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %stack2 = forth.call @dub2(%stack1)
           : (tensor<*xi32>) -> tensor<*xi32>
    ```

    This is only valid if a function named "dub2" exists and takes one argument.
  }];

  // The call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, I32Tensor:$stack);

  // The call operation returns a single value of TensorType or
  // StructType.
  let results = (outs I32Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $stack `)` attr-dict `:` functional-type($stack, results)
  }];

  // Add custom build methods for the call operation.
  let builders = [
    OpBuilder<(ins "mlir::StringRef":$callee, "mlir::Value":$stack)>
  ];
}
