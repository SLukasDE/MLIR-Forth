//===----------------------------------------------------------------------===//
// WriteCharOp
//===----------------------------------------------------------------------===//

def WriteCharOp : ForthOp<"writeChar", [Pure]>
{
  let summary = " Operation to print a character to standard output ";
  
  let description = [{
    Prints a character
  }];
  
  let arguments = (ins I32:$character);

  let results = (outs);

  //let builders = [
  //  OpBuilder<(ins "mlir::Value":$character)>
  //];
}
