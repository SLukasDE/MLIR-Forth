#include "forth/ir1/ForthDialect.hpp"
#include "forth/ir1/Ops.hpp"



#include "forth/ir1/ForthDialect.cpp.inc"

namespace forth {
namespace ir1 {

void ForthDialect::initialize()
{
	addOperations<
#define GET_OP_LIST
#include "forth/ir1/Ops.cpp.inc"
                >();
}

} // namespace ir1
} // namespace forth
