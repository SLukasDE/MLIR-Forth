#ifndef FORTH_MAIN_H_
#define FORTH_MAIN_H_

#include "forth/lang/Node.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"


#include <memory>
#include <string>

namespace forth {

class Main {
public:
	Main(int argc, char *argv[]);

	mlir::LogicalResult run();

private:
	enum InputType {
		Forth,
		MLIR,
		Unknown
	};

	enum ActionType {
		None,
		DumpAST,
		DumpMLIR
	};

	llvm::cl::opt<std::string> inputFilename;
	llvm::cl::opt<enum InputType> inputType;
	llvm::cl::opt<enum ActionType> emitAction;

	std::unique_ptr<forth::lang::Block> forthAST;

	mlir::DialectRegistry registry;
	mlir::MLIRContext context;
	mlir::OwningOpRef<mlir::ModuleOp> module;

	mlir::LogicalResult dumpAST();
	mlir::LogicalResult dumpMLIR();
	mlir::LogicalResult loadFileForth();
	mlir::LogicalResult loadFileMLIR();
};

} /* namespace forth */

#endif /* FORTH_MAIN_H_ */
