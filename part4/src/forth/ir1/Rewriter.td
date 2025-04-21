include "Ops.td"

include "mlir/IR/PatternBase.td"

def createTFLLeakyRelu : NativeCodeCall<"createTFLLeakyRelu($_builder, $0.getDefiningOp(), $1, $2)">;

//def : Pat<(ForthDialect_AddOp:$old_value, $arg, F32Attr:$a),
//def : Pat<(AddOp:$old_value $arg1),
//          (createTFLLeakyRelu $old_value, $arg1)>;
