include "ops/ForthOp.td"

include "ops/PickOp.td" // dynamic
include "ops/GetOp.td"  // static
include "ops/PopOp.td" // static
include "ops/PushOp.td"     // dynamic
include "ops/ConstantOp.td" // static

include "ops/StoreOp.td"
include "ops/LoadOp.td"

include "ops/ExecuteOp.td" // dynamic, by address/number
include "ops/CallOp.td"    // static, by name
include "ops/FuncOp.td"
include "ops/ReturnOp.td"
include "ops/StackOp.td"

include "ops/ReadCharOp.td"     // dynamic
include "ops/WriteCharOp.td"    // dynamic, by char
include "ops/WriteIntegerOp.td" // dynamic, by int
include "ops/WriteStringOp.td"  // static, by string

include "ops/AddOp.td"
include "ops/DivOp.td"
include "ops/SubOp.td"
include "ops/MulOp.td"

