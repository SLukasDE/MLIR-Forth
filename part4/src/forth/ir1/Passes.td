include "ForthDialect.td"

def ConvertForth2Arith : Pass<"convert-forth-to-arith", "mlir::ModuleOp">{
  let summary = "Convert forth-dialect operations to arith-dialect";
  
  // let description = [{
  //   "..."
  // }];
  
  let dependentDialects = ["forth::ir1::ForthDialect"];
}
