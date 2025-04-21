//===----------------------------------------------------------------------===//
// ReadCharOp
//===----------------------------------------------------------------------===//

def ReadCharOp : ForthOp<"readChar", [Pure]>
{
  let summary = " Operation to read a character to standard input ";
  
  let description = [{
    Reads a character
  }];
  
  let arguments = (ins);

  let results = (outs I32:$character);

  let builders = [
    OpBuilder<(ins)>
  ];
}
