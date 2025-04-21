#include "forth/ir1/Ops.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"



#define GET_OP_CLASSES
#include "forth/ir1/Ops.cpp.inc"

namespace forth {
namespace ir1 {


//===----------------------------------------------------------------------===//
// PickOp
//===----------------------------------------------------------------------===//

void PickOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack, mlir::Value n) {
	// definition of return type
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);
	state.addOperands(n);
}


//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

void GetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack, unsigned int n) {
	// definition of return type
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);

	mlir::IntegerType attributeType = builder.getIntegerType(32, /*isSigned=*/false);
	auto nAttr = mlir::IntegerAttr::get(attributeType, n);
	state.addAttribute("n", nAttr);
}


//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

void PopOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack, unsigned int n) {
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);
	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);

	mlir::IntegerType attributeType = builder.getIntegerType(32, /*isSigned=*/false);
	auto nAttr = mlir::IntegerAttr::get(attributeType, n);
	state.addAttribute("n", nAttr);
}


//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

void PushOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack, mlir::Value value) {
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);
	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operands
	state.addOperands(stack);
	state.addOperands(value);
}


//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, int value) {
	//ConstantOp::build(builder, state, stack, value);
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);
	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operand
	state.addAttribute("value", builder.getI32IntegerAttr(value));
}

//############################################################################//

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack) {
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);

	int64_t s = -1;
	if(mlir::ShapedType::isDynamic(s)) {
	}

	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);
}


//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack) {
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);
	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);
}

//############################################################################//

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

void ExecuteOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value stack, mlir::Value addr) {
	// definition of return type
	llvm::SmallVector<int64_t> dynamicShape(1, mlir::ShapedType::kDynamic);
	mlir::Type elementType = builder.getI32Type();
	mlir::Type type = mlir::RankedTensorType::get(dynamicShape, elementType, nullptr);
	state.addTypes(type);

	// definition of operand
	state.addOperands(stack);
	state.addOperands(addr);
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef callee, mlir::Value stack) {
	// definiert den Return-Type
	llvm::ArrayRef<int64_t> typeShape{mlir::ShapedType::kDynamic};
	/// Finally we build a tensor type from a list of shaped dimensions, that is 'tensor<?xi32>'
	mlir::Type type = mlir::RankedTensorType::get(typeShape, builder.getI32Type());
	if (!type)
		// error
		return;
	state.addTypes(type);

	// definiert einen Operand (Aufruf-Argument)
	state.addOperands(stack);

	// definiert ein Attribut (Aufruf-Argument)
	state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the call interface.
mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
	return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call interface.
void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
	(*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the call interface.
mlir::Operation::operand_range CallOp::getArgOperands() {
	//	return getStack();

	// mlir::TypedValue<mlir::TensorType> typedTensorValue = getStack();
	// mlir::Value value = typedTensorValue;

#if 1
	mlir::OpOperand& opOperand = getStackMutable();
	llvm::detail::indexed_accessor_range_base<mlir::OperandRange, mlir::OpOperand*, mlir::Value, mlir::Value, mlir::Value> iarb(&opOperand, 1);
	mlir::OperandRange operandRange(iarb);
#else
	mlir::Operation* operation = getOperation();
	mlir::OperandRange operandTange = operation->getOperands();
#endif

	return operandRange;
}

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
mlir::MutableOperandRange CallOp::getArgOperandsMutable() {
	//return getStackMutable();

	mlir::OpOperand& opOperand = getStackMutable();
#if 1
	mlir::MutableOperandRange mutableOperands(opOperand);
#else
	mlir::Operation* operation = getOperation();
	mlir::MutableOperandRange mutableOperands(operation);
	mutableOperands.append(opOperand.get());  // Füge neuen Operand hinzu
#endif
	return mutableOperands;
}


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name) {
	llvm::SmallVector<int64_t> typeShape(1, mlir::ShapedType::kDynamic);
	/// Finally we build a tensor type from a list of shaped dimensions, that is 'tensor<?xi32>'
	mlir::Type argType = mlir::RankedTensorType::get(typeShape, builder.getI32Type());
	mlir::Type resultType = mlir::RankedTensorType::get(typeShape, builder.getI32Type());

	// Funktionssignatur erstellen
	mlir::FunctionType functionType = builder.getFunctionType(
	    /*inputs=*/{argType},
	    /*results=*/{resultType}
	);

	build(builder, state, name, functionType);
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name, mlir::FunctionType functionType, mlir::ArrayRef<mlir::NamedAttribute> attrs) {
	// FunctionOpInterface provides a convenient `build` method that will populate
	// the state of our FuncOp, and create an entry block.
	buildWithEntryBlock(builder, state, name, functionType, attrs, functionType.getInputs());
}

llvm::LogicalResult FuncOp::verify() {
	mlir::FunctionType functionType = getFunctionType();

	if(getSymName() == "main") {
		if (functionType.getNumInputs() != 0) {
			return emitError("Erwarte kein Argument");
		}

		if (functionType.getNumResults() != 0) {
			return emitError("Erwarte kein Rückgabewert");
		}
	}
	else {
		mlir::Type type;

		if (functionType.getNumInputs() != 1) {
			return emitError("Erwarte genau ein Argument");
		}

		type = functionType.getInput(0);
		if (!mlir::isa<mlir::RankedTensorType>(type)) {
			return emitError("Erwarte ein Tensor-Argument");
		}

		if (functionType.getNumResults() != 1) {
			return emitError("Erwarte genau einen Rückgabewert");
		}

		type = functionType.getResult(0);
		if (!llvm::isa<mlir::RankedTensorType>(type)) {
			return emitError("Erwarte einen Tensor-Rückgabewert");
		}
	}

	return llvm::success();
}


//===----------------------------------------------------------------------===//
// StackOp
//===----------------------------------------------------------------------===//

void StackOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, unsigned int stackSize) {
	// definiert den Return-Type
	llvm::ArrayRef<int64_t> typeShape{mlir::ShapedType::kDynamic};
	/// Finally we build a tensor type from a list of shaped dimensions, that is 'tensor<?xi32>'
	mlir::Type type = mlir::RankedTensorType::get(typeShape, builder.getI32Type());

	if (!type) {
		// error
		return;
	}
	state.addTypes(type);


	mlir::IntegerType attributeType = builder.getIntegerType(32, /*isSigned=*/false);
	auto stackSizeAttr = mlir::IntegerAttr::get(attributeType, stackSize);
	state.addAttribute("stackSize", stackSizeAttr);
}

//############################################################################//

//===----------------------------------------------------------------------===//
// ReadCharOp
//===----------------------------------------------------------------------===//
void ReadCharOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
	state.addTypes(builder.getI32Type());
}

//############################################################################//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand1, mlir::Value operand2) {
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(operand1);
	state.addOperands(operand2);
}
/*
// static
void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet&, mlir::MLIRContext*) {

}

llvm::LogicalResult AddOp::verify() {

}

llvm::LogicalResult AddOp::verifyRegions() {

}
*/


//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand1, mlir::Value operand2) {
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(operand1);
	state.addOperands(operand2);
}


//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand1, mlir::Value operand2) {
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(operand1);
	state.addOperands(operand2);
}


//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand1, mlir::Value operand2) {
	mlir::Type type = builder.getI32Type();
	state.addTypes(type);

	// definition of operand
	state.addOperands(operand1);
	state.addOperands(operand2);
}

} // namespace ir1
} // namespace forth
