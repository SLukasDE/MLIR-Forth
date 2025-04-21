#ifndef MLIR_FORTH_IR1_PASSES_
#define MLIR_FORTH_IR1_PASSES_

#include "forth/ir1/ForthDialect.hpp"
#include "mlir/Conversion/Passes.h"

namespace forth {
namespace ir1 {

#define GEN_PASS_DECL_CONVERTFORTH2ARITH
#include "forth/ir1/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "forth/ir1/Passes.h.inc"

#define GEN_PASS_DECL_CONVERTFORTH2ARITH
#include "forth/ir1/Passes.h.inc"

} // namespace ir1
} // namespace forth


#endif /* MLIR_FORTH_IR1_PASSES_ */
