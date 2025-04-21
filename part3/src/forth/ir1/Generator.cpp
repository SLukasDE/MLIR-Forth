#include "forth/ir1/Generator.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


namespace forth {
namespace ir1 {

/// convert the AST for a Forth module (source file) to an MLIR Module operation.
mlir::LogicalResult Generator::generate(mlir::OpBuilder& builder, mlir::ModuleOp module, const lang::Block &ast) {
	return Generator(builder, module, ast).generatorResult;
}

Generator::Generator(mlir::OpBuilder& aBuilder, mlir::ModuleOp& aModule, const lang::Block& moduleAST)
: builder(aBuilder),
  module(aModule),
  generatorResult(mlir::failure())
{
	// create all user defined variables
	for (auto &node : moduleAST) {
		if (lang::NodeVariable* nodeVariable = llvm::dyn_cast<lang::NodeVariable>(node.get())) {
			if(mlir::failed(createVariable(*nodeVariable))) {
				return;
			}
		}
	}

	// create all builtin words as function
	if(mlir::failed(createFunctionBuiltin_DUP())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_DROP())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_OVER())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_PICK())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_ROT())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_SWAP())) {
		return;
	}

	if(mlir::failed(createFunctionBuiltin_LOAD())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_STORE())) {
		return;
	}

	if(mlir::failed(createFunctionBuiltin_EXECUTE())) {
		return;
	}

	if(mlir::failed(createFunctionBuiltin_EMIT())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_DOT())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_CR())) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_KEY())) {
		return;
	}

	if(mlir::failed(createFunctionBuiltin_MATH<ir1::AddOp>("+"))) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_MATH<ir1::SubOp>("-"))) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_MATH<ir1::MulOp>("*"))) {
		return;
	}
	if(mlir::failed(createFunctionBuiltin_MATH<ir1::DivOp>("/"))) {
		return;
	}

	// create all user defined functions/words
	for (auto &node : moduleAST) {
		if (lang::NodeProcedure *nodeProcedure = llvm::dyn_cast<lang::NodeProcedure>(node.get())) {
			if(mlir::failed(createFunctionUserDefined(*nodeProcedure))) {
				return;
			}
		}
	}

	llvm::SmallVector<mlir::Type> argumentTypes;
	llvm::SmallVector<mlir::Type> resultTypes;

	// Create an MLIR function for the given prototype.
	builder.setInsertionPointToEnd(module.getBody());
	//auto funcType = builder.getFunctionType(argumentTypes, resultTypes);

	mlir::Location currentLocation = builder.getUnknownLoc();
	generatorResult = createFunction("main", currentLocation, [&](mlir::Value) {
		ir1::StackOp stackOp = builder.create<ir1::StackOp>(currentLocation, 65535);
		mlir::Operation *stackOperation = stackOp.getOperation();
		if(!stackOperation) {
			// error
			return mlir::Value();
		}
		mlir::OpResult opResult = stackOperation->getResult(0);
		mlir::Value stack = opResult;

		// Emit the body of the function.
		stack = createNodeList(stack, moduleAST, currentLocation);

		return stack;
	} );

	if(mlir::failed(generatorResult)) {
		mlir::emitError(builder.getUnknownLoc(), "Error recognized in function 'generate'");
	}
}

mlir::LogicalResult Generator::createVariable(const lang::NodeVariable &nodeVariable) {
	// Typ der globalen Variable (i32)
	mlir::Type type = builder.getI32Type();

	// Name und Initialwert
	llvm::StringRef name = nodeVariable.name;
	mlir::Attribute initValue = builder.getI32IntegerAttr(0);

	// Operation erstellen
	mlir::LLVM::GlobalOp globalOp = builder.create<mlir::LLVM::GlobalOp>(
			convertLocation(nodeVariable.location),
			type,
			/*isConstant=*/false,  // mutable
			mlir::LLVM::Linkage::Internal,
			name,
			initValue
	);
	if (!globalOp) {
		emitError(nodeVariable.location, "error");
		return mlir::failure();
	}

	// Global zur ModuleOp hinzufügen
	module.push_back(globalOp);

	// Add the procedure into into the symbol table.
	if(indexBySymbol.insert(std::make_pair(nodeVariable.name, std::make_pair(symbolByIndex.size(), lang::Node::NodeKind::variable))).second == false) {
		return mlir::failure();
	}
	symbolByIndex.emplace_back(nodeVariable.name);

	return mlir::success();
}

mlir::LogicalResult Generator::createFunctionUserDefined(const lang::NodeProcedure &nodeProcedure) {
	mlir::Location currentLocation = convertLocation(nodeProcedure.location);

	return createFunction(nodeProcedure.name, currentLocation, [&](mlir::Value s) {
		return createNodeList(s, nodeProcedure.body, currentLocation);
	} );
}

template <typename T>
mlir::LogicalResult Generator::createFunctionBuiltin(const std::string& functionName) {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction(functionName, currentLocation, [&](mlir::Value s) {
		T op = builder.create<T>(currentLocation, s);
		if(!op) {
			// error
			return mlir::Value();
		}

		mlir::Operation* operation = op.getOperation();
		if(!operation) {
			// error
			return mlir::Value();
		}

		s = operation->getResult(0);
		return s;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_DROP() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("DROP", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_DUP() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("DUP", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);

		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, value);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_OVER() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("OVER", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 1);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);

		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, value);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_PICK() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("PICK", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value n = operation->getResult(0);

		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		ir1::PickOp pickOp = builder.create<ir1::PickOp>(currentLocation, stack, n);
		operation = pickOp ? pickOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);

		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, value);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_ROT() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("ROT", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp0 = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp0 ? getOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v0 = operation->getResult(0);

		ir1::GetOp getOp1 = builder.create<ir1::GetOp>(currentLocation, stack, 1);
		operation = getOp1 ? getOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v1 = operation->getResult(0);

		ir1::GetOp getOp2 = builder.create<ir1::GetOp>(currentLocation, stack, 2);
		operation = getOp2 ? getOp2.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v2 = operation->getResult(0);



		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 3);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);



		ir1::PushOp pushOp0 = builder.create<ir1::PushOp>(currentLocation, stack, v1);
		operation = pushOp0 ? pushOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		ir1::PushOp pushOp1 = builder.create<ir1::PushOp>(currentLocation, stack, v2);
		operation = pushOp1 ? pushOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		ir1::PushOp pushOp2 = builder.create<ir1::PushOp>(currentLocation, stack, v0);
		operation = pushOp2 ? pushOp2.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_SWAP() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("SWAP", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp0 = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp0 ? getOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v0 = operation->getResult(0);

		ir1::GetOp getOp1 = builder.create<ir1::GetOp>(currentLocation, stack, 1);
		operation = getOp1 ? getOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v1 = operation->getResult(0);



		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 2);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);



		ir1::PushOp pushOp0 = builder.create<ir1::PushOp>(currentLocation, stack, v0);
		operation = pushOp0 ? pushOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		ir1::PushOp pushOp1 = builder.create<ir1::PushOp>(currentLocation, stack, v1);
		operation = pushOp1 ? pushOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_LOAD() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("@", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value address = operation->getResult(0);


		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);


		// Wert laden
		mlir::Type type = builder.getI32Type();
		mlir::LLVM::LoadOp loadOp = builder.create<mlir::LLVM::LoadOp>(currentLocation, type, address);
		operation = loadOp ? loadOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value loadedValue = operation->getResult(0);


		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, loadedValue);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_STORE() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("!", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp0 = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp0 ? getOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);


		ir1::GetOp getOp1 = builder.create<ir1::GetOp>(currentLocation, stack, 1);
		operation = getOp1 ? getOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value address = operation->getResult(0);


		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 2);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);


		// Wert speichern
		builder.create<mlir::LLVM::StoreOp>(currentLocation, value, address);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_EXECUTE() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("EXECUTE", currentLocation, [&](mlir::Value stack) {
#if 1
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value addr = operation->getResult(0);
#else
		addr = getOp.getResult();
#endif


		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
#if 0
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);
#else
		stack = popOp.getResult();
#endif


		ir1::ExecuteOp executeOp = builder.create<ir1::ExecuteOp>(currentLocation, stack, addr);
#if 0
		operation = executeOp ? executeOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);
#else
		stack = executeOp.getResult();
#endif

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_EMIT() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("EMIT", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);


		builder.create<ir1::WriteCharOp>(currentLocation, value);


		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_DOT() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction(".", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp ? getOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);


		builder.create<ir1::WriteIntegerOp>(currentLocation, value);


		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 1);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_CR() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("CR", currentLocation, [&](mlir::Value stack) {
		if(mlir::failed(createEmitString(currentLocation, "\n"))) {
			// error
			return mlir::Value();
		}

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunctionBuiltin_KEY() {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction("KEY", currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::ReadCharOp readCharOp = builder.create<ir1::ReadCharOp>(currentLocation);
		operation = readCharOp ? readCharOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value value = operation->getResult(0);


		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, value);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

template<typename T>
mlir::LogicalResult Generator::createFunctionBuiltin_MATH(const std::string& functionName) {
	mlir::Location currentLocation = builder.getUnknownLoc();

	return createFunction(functionName, currentLocation, [&](mlir::Value stack) {
		mlir::Operation* operation;

		ir1::GetOp getOp0 = builder.create<ir1::GetOp>(currentLocation, stack, 0);
		operation = getOp0 ? getOp0.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v0 = operation->getResult(0);

		ir1::GetOp getOp1 = builder.create<ir1::GetOp>(currentLocation, stack, 1);
		operation = getOp1 ? getOp1.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value v1 = operation->getResult(0);



		ir1::PopOp popOp = builder.create<ir1::PopOp>(currentLocation, stack, 2);
		operation = popOp ? popOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);



		T mathOp = builder.create<T>(currentLocation, v0, v1);
		operation = mathOp ? mathOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		mlir::Value result = operation->getResult(0);



		ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, result);
		operation = pushOp ? pushOp.getOperation() : nullptr;
		if(!operation) {
			// error
			return mlir::Value();
		}
		stack = operation->getResult(0);

		return stack;
	} );
}

mlir::LogicalResult Generator::createFunction(const std::string& funcName, mlir::Location currentLocation, std::function<mlir::Value(mlir::Value)> createBody) {
	// Build a list of sizes for each dimension to shape a tensor.
	// In this case, the is only one dimension, so we build a tensor with rank 1.
	// The size of the 1st dimension is -1, that means the size is unknown
	llvm::ArrayRef<int64_t> typeShape{mlir::ShapedType::kDynamic};
	/// Finally we build a tensor type from a list of shaped dimensions, that is 'tensor<?xi32>'
	mlir::Type type = mlir::RankedTensorType::get(typeShape, builder.getI32Type());
	if (!type) {
		return mlir::failure();
	}

	llvm::SmallVector<mlir::Type> argumentTypes;
	llvm::SmallVector<mlir::Type> resultTypes;

	if (funcName != "main") {
		argumentTypes.push_back(type);
		resultTypes.push_back(type);
	}

	// Create an MLIR function for the given prototype.

	builder.setInsertionPointToEnd(module.getBody());
	mlir::FunctionType functionType = builder.getFunctionType(argumentTypes, resultTypes);
	ir1::FuncOp funcOp = builder.create<ir1::FuncOp>(currentLocation, funcName, functionType);
	if (!funcOp) {
		mlir::emitError(currentLocation, "error");
		return mlir::failure();
	}

	// If this function isn't main, then set the visibility to private.
	if (funcName != "main") {
		funcOp.setPrivate();
	}

	mlir::Block& entryBlock = funcOp.getBody().front();
	mlir::Value stack;
	if (funcName != "main") {
		stack = entryBlock.getArgument(0);
	}

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Emit the body of the function.
	stack = createBody(stack);
	if (!stack) {
	//if (!stack && funcName != "main") {
		funcOp.erase();
		mlir::emitError(currentLocation, "Error recognized in function 'createFunction' when creating funktion '" + funcName + "'");
		return mlir::failure();
	}

	// Add the procedure into into the symbol table.
	if(indexBySymbol.insert(std::make_pair(funcName, std::make_pair(symbolByIndex.size(), lang::Node::NodeKind::procedure))).second == false) {
		return mlir::failure();
	}
	symbolByIndex.emplace_back(funcName);

	/// Emit a return operation.
	builder.create<ir1::ReturnOp>(currentLocation, stack ? llvm::ArrayRef(stack) : llvm::ArrayRef<mlir::Value>());

	return mlir::success();
}

mlir::LogicalResult Generator::createEmitString(mlir::Location location, const std::string& text) {
	ir1::WriteStringOp writeStringOp = builder.create<ir1::WriteStringOp>(location, text);
	if (!writeStringOp) {
		return mlir::failure();
	}
	return mlir::success();
}

/// Codegen a list of words, return empty stack if one of them hit an error.
mlir::Value Generator::createNodeList(mlir::Value stack, const std::vector<std::unique_ptr<lang::Node>> &expressionList, mlir::Location& currentLocation) {
	for (const auto &node : expressionList) {
		if(mlir::failed(createNode(stack, *node.get(), currentLocation))) {
			// error
			emitError(node.get()->location, "Error recognized in function 'createNodeList'");
			return nullptr;
		}
	}

	return stack;
}

mlir::LogicalResult Generator::createNode(mlir::Value& stack, const lang::Node &node, mlir::Location& currentLocation) {
	currentLocation = convertLocation(node.location);

	switch(node.getKind()) {
	case lang::Node::NodeKind::comment:
	case lang::Node::NodeKind::procedure:
	case lang::Node::NodeKind::variable:
		break;
	case lang::Node::NodeKind::number:
		stack = createNumber(stack, llvm::cast<lang::NodeNumber>(node).number, convertLocation(node.location));
		if(!stack) {
			emitError(node.location, "Error recognized in function 'createNode' after calling 'createNumber'");
			return mlir::failure();
		}
		break;
	case lang::Node::NodeKind::word:
		stack = callWord(stack, llvm::cast<lang::NodeWord>(node));
		if(!stack) {
			emitError(node.location, "Error recognized in function 'createNode' after calling 'callWord'");
			return mlir::failure();
		}
		break;
	case lang::Node::NodeKind::addrOfWord:
		stack = createAddressOfWord(stack, llvm::cast<lang::NodeAddrOfWord>(node));
		if(!stack) {
			emitError(node.location, "Error recognized in function 'createNode' after calling 'createAddressOfWord'");
			return mlir::failure();
		}
		break;
	case lang::Node::NodeKind::emitString:
		if(mlir::failed(createEmitString(convertLocation(node.location), llvm::cast<lang::NodeEmitString>(node).str))) {
			emitError(node.location, "Error recognized in function 'createNode' after calling 'createEmitString'");
			// error
			return mlir::failure();
		}
		return mlir::success();
	default:
		// unkown kind. Cancel or continue?
		emitError(node.location, "Unknown kind of node in 'createNode'");
		return mlir::failure();
	}

	return mlir::success();
}

mlir::Value Generator::createNumber(mlir::Value stack, int number, mlir::Location location) {
	mlir::Operation *operation;

	ir1::ConstantOp constantOp = builder.create<ir1::ConstantOp>(location, number);
	operation = constantOp ? constantOp.getOperation() : nullptr;
	if(!operation) {
		// error
		return mlir::Value();
	}
	mlir::Value value = operation->getResult(0);

	ir1::PushOp pushOp = builder.create<ir1::PushOp>(location, stack, value);
	operation = pushOp ? pushOp.getOperation() : nullptr;
	if(!operation) {
		// error
		return mlir::Value();
	}
	mlir::Value returnStack = operation->getResult(0);

	return returnStack;
}

mlir::Value Generator::callWord(mlir::Value stack, const lang::NodeWord &nodeWord) {
	auto iter = indexBySymbol.find(nodeWord.word);
	if(iter == std::end(indexBySymbol)){
		// error
		emitError(nodeWord.location, "Requested word \"" + nodeWord.word + "\" does not exists");
		return nullptr;
	}

	mlir::Value returnStack;

	if(iter->second.second == lang::Node::NodeKind::procedure) {
		mlir::Operation* operation;

		ir1::CallOp callOp = builder.create<ir1::CallOp>(convertLocation(nodeWord.location), nodeWord.word, stack);
		operation = callOp ? callOp.getOperation() : nullptr;
		if(operation) {
			returnStack = operation->getResult(0);
		}
	}
	else if(iter->second.second == lang::Node::NodeKind::variable) {
		mlir::Operation* operation;

		mlir::Location currentLocation = builder.getUnknownLoc();
		mlir::Type type = builder.getI32Type();

		// Adresse der Globalen holen
		mlir::LLVM::AddressOfOp addressOp = builder.create<mlir::LLVM::AddressOfOp>(
				currentLocation,
				//mlir::LLVM::LLVMPointerType::get(type),
				//mlir::LLVM::LLVMPointerType::get(builder.getContext()),
				type,
				nodeWord.word);

		operation = addressOp ? addressOp.getOperation() : nullptr;
		if(operation) {
			mlir::Value value = operation->getResult(0);

			ir1::PushOp pushOp = builder.create<ir1::PushOp>(currentLocation, stack, value);
			operation = pushOp ? pushOp.getOperation() : nullptr;
			if(operation) {
				returnStack = operation->getResult(0);
			}
		}

	}

	if(!returnStack) {
		// error
	}

	return returnStack;
}

mlir::Value Generator::createAddressOfWord(mlir::Value stack, const lang::NodeAddrOfWord &nodeAddrOfWord) {
	auto iter = indexBySymbol.find(nodeAddrOfWord.word);

	if(iter == std::end(indexBySymbol)){
		// error
		emitError(nodeAddrOfWord.location, "Requested word \"" + nodeAddrOfWord.word + "\" does not exists");
		return nullptr;
	}

	return createNumber(stack, iter->second.first, convertLocation(nodeAddrOfWord.location));
}

/// Helper conversion for a Forth AST location to an MLIR location.
mlir::Location  Generator::convertLocation(const lang::Location &location) {
	if(location.line <= 0 || location.col <= 0) {
		return builder.getUnknownLoc();
	}
	return mlir::FileLineColLoc::get(builder.getStringAttr(location.file ? *location.file : ""), location.line, location.col);
}

void Generator::emitError(const lang::Location &location, llvm::StringRef message) {
	//const Twine &message
	mlir::emitError(convertLocation(location), message);
}

} /* namespace ir1 */
} /* namespace forth */
