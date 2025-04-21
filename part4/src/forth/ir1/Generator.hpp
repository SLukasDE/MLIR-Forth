#ifndef FORTH_IR1_GENERATOR_H_
#define FORTH_IR1_GENERATOR_H_

#include "forth/ir1/Ops.hpp"
#include "forth/lang/Node.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>


namespace forth {
namespace ir1 {

class Generator {
public:
	/// Emit IR for the given Forth AST (block), returns success or failure.
	static mlir::LogicalResult generate(mlir::OpBuilder& builder, mlir::ModuleOp module, const lang::Block &moduleAST);

private:
	/// Stateful helper class to create IR. Stateful means in the terms of "insertion points", where to put the next operation.
	mlir::OpBuilder& builder;

	/// A "module" matches a forth source file
	mlir::ModuleOp& module;

	/// was generating the code successful
	mlir::LogicalResult generatorResult;

	/// The symbol table maps a symbol name of a variable or a procedure to an index and the kind of the symbol name (variable or procedure).
	std::map<std::string, std::pair<std::size_t, lang::Node::NodeKind>> indexBySymbol;
	std::vector<std::string> symbolByIndex;

	Generator(mlir::OpBuilder& builder, mlir::ModuleOp& module, const lang::Block& moduleAST);

	mlir::LogicalResult createVariable(const lang::NodeVariable& nodeVariable);
	template<typename T>
	mlir::LogicalResult createFunctionBuiltin(const std::string& functionName);
	mlir::LogicalResult createFunctionBuiltin_DROP();
	mlir::LogicalResult createFunctionBuiltin_DUP();
	mlir::LogicalResult createFunctionBuiltin_OVER();
	mlir::LogicalResult createFunctionBuiltin_PICK();
	mlir::LogicalResult createFunctionBuiltin_ROT();
	mlir::LogicalResult createFunctionBuiltin_SWAP();

	mlir::LogicalResult createFunctionBuiltin_LOAD();
	mlir::LogicalResult createFunctionBuiltin_STORE();

	mlir::LogicalResult createFunctionBuiltin_EXECUTE();

	mlir::LogicalResult createFunctionBuiltin_EMIT();
	mlir::LogicalResult createFunctionBuiltin_DOT();
	mlir::LogicalResult createFunctionBuiltin_CR();
	mlir::LogicalResult createFunctionBuiltin_KEY();

	template<typename T>
	mlir::LogicalResult createFunctionBuiltin_MATH(const std::string& functionName);

	mlir::LogicalResult createFunctionUserDefined(const lang::NodeProcedure& nodeProcedure);
	mlir::LogicalResult createFunction(const std::string& functionName, mlir::Location location, std::function<mlir::Value(mlir::Value)> createBody);
	mlir::LogicalResult createEmitString(mlir::Location location, const std::string& text);

	mlir::Value createNodeList(mlir::Value stack, const std::vector<std::unique_ptr<lang::Node>>& expressionList, mlir::Location& location);
	mlir::LogicalResult createNode(mlir::Value& stack, const lang::Node &node, mlir::Location& location);
	mlir::Value createNumber(mlir::Value stack, int number, mlir::Location location);
	mlir::Value callWord(mlir::Value stack, const lang::NodeWord& nodeWord);
	mlir::Value createAddressOfWord(mlir::Value stack, const lang::NodeAddrOfWord& nodeAddrOfWord);

	mlir::Location convertLocation(const lang::Location& location);
	void emitError(const lang::Location& location, llvm::StringRef message);
};

} /* namespace ir1 */
} /* namespace forth */

#endif /* FORTH_IR1_GENERATOR_H_ */
