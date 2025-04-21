
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/FunctionInterfaces.td"
//include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ForthDialect
//===----------------------------------------------------------------------===//

def ForthDialect : Dialect
{
  let summary = "A sample dialect for understanding";
  
  let name = "forth";
  let cppNamespace = "forth::ir1";
}
