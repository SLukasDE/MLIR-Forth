//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : ForthOp<"constant", [Pure]>
{
  let summary = "Pushs a number on the stack"; 
  
  // let description = [{
  //   ...
  // }];
  
  let arguments = (ins I32Attr:$value);

  let results = (outs I32:$output);
  
  let assemblyFormat = "$value attr-dict `:` type($output)";
  
  let builders = [
    OpBuilder<(ins "int":$value)>
  ];
  
  //let hasCustomAssemblyFormat = 1;
  
}
