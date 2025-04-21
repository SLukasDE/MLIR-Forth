//===----------------------------------------------------------------------===//
// WriteIntegerOp
//===----------------------------------------------------------------------===//

def WriteIntegerOp : ForthOp<"writeInteger", [Pure]>
{
  let summary = " Operation to print an integer to standard output ";
  
  let description = [{
    Prints an integer
  }];
  
  let arguments = (ins I32:$integer);

  let results = (outs);

  //let builders = [
  //  OpBuilder<(ins "mlir::Value":$integer)>
  //];
}
