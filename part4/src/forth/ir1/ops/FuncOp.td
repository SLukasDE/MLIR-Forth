//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : ForthOp<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "user defined word";
  
  let description = [{
  
    The "forth.func" operation represents a user defined word by writing
    ```PROCEDURE <word> : ... ; ``` in FORTH language. 

    Example:

    ```mlir
    forth.func @dub2(%stack0: tensor<?xi32>) -> tensor<?xi32> {
      %stack1 = forth.dub %stack0 : (tensor<?xi32>) -> tensor<?xi32>
      %stack2 = forth.dub %stack1 : (tensor<?xi32>) -> tensor<?xi32>
      return %stack3 : tensor<?xi32> 
    }
    
    forth.func @main() -> () {
      %stack0 = forth.stack 65535 : (ui32) -> tensor<?xi32>
      %stack1 = forth.constant %stack0, 10 : (tensor<?xi32>, i32) -> tensor<?xi32> 
      %stack2 = forth.constant %stack1, 20 : (tensor<?xi32>, i32) -> tensor<?xi32> 
      %stack3 = forth.add %stack2 : (tensor<?xi32>) -> tensor<?xi32>
      return %stack4 : tensor<?xi32> 
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  
  //let results = (outs);
  
  
  // ":$body" sorgt dafür, dass für die Region automatisch ein Getter "getBody()" erstellt wird 
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::StringRef":$name)>,
    OpBuilder<(ins "mlir::StringRef":$name, "mlir::FunctionType":$functionType, CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>
  ];
  
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];
  
  //let assemblyFormat = [{
  //  $sym_name `(` `)` `->` type($function_type) attr-dict-with-keyword
  //}];
  //let hasCustomAssemblyFormat = 1;
  
  let hasVerifier = 1;
}
