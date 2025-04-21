#include "forth/ir1/Passes.hpp"
#include "forth/ir1/Ops.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace forth {
namespace ir1 {

#define GEN_PASS_DEF_CONVERTFORTH2ARITH
#include "forth/ir1/Passes.h.inc"


struct ConvertForth2Arith : public forth::ir1::impl::ConvertForth2ArithBase<ConvertForth2Arith>
{
	using ConvertForth2ArithBase::ConvertForth2ArithBase;
	void runOnOperation() override;
};

void ConvertForth2Arith::runOnOperation()
{
	mlir::Operation& operation = *getOperation();

	llvm::outs() << "OpName: \"" << operation.getName() << "\"\n";

	operation.walk([&](forth::ir1::ConstantOp constOp) {
		llvm::outs() << "CHECK 0\n";
		mlir::OpBuilder opBuilder(constOp);

		llvm::outs() << "ConstOp-Name: \"" << constOp.getOperationName() << "\"\n";
		mlir::arith::ConstantOp newOp = opBuilder.create<mlir::arith::ConstantOp>(opBuilder.getUnknownLoc(), opBuilder.getI32Type(), opBuilder.getI32IntegerAttr(33));
		//auto newOp = opBuilder.create<mlir::arith::ConstantOp>(opBuilder.getUnknownLoc(), opBuilder.getI32Type(), opBuilder.getI32IntegerAttr(33));
		llvm::outs() << "NewOp-Name: \"" << newOp.getOperationName() << "\"\n";

		//constOp->replaceAllUsesWith(newOp);
		llvm::outs() << "CHECK 1\n";
		constOp->erase();
		llvm::outs() << "CHECK 2\n";
	});
}

} // namespace ir1
} // namespace forth

