//===----------------------------------------------------------------------===//
// WriteStringOp
//===----------------------------------------------------------------------===//

def WriteStringOp : ForthOp<"writeString", [Pure]>
{
  let summary = [{ operation to print a string to standard output }];
  
  let description = [{
    Prints a string
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$text);
  let results = (outs);
}
