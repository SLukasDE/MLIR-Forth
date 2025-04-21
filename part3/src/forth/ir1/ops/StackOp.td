//===----------------------------------------------------------------------===//
// StackOp
//===----------------------------------------------------------------------===//

def StackOp : ForthOp<"stack", [Pure]>
{
  let summary = "return a stack to operate with forth-words"; 
  
  // let description = [{
  //   ...
  // }];
  
  let arguments = (UI32Attr:$stackSize);
  let results = (outs I32Tensor);
  
  let builders = [
    OpBuilder<(ins "unsigned int":$stackSize)>
  ];
  
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1 ;
}
