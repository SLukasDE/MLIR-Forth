#include "forth/Main.hpp"
#include "forth/lang/Lexer.hpp"
#include "forth/lang/Parser.hpp"
#include "forth/lang/Test.hpp"
#include "forth/ir1/Generator.hpp"
#include "forth/ir1/ForthDialect.hpp"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <fstream>
#include <string_view>

namespace forth {

Main::Main(int argc, char *argv[])
: inputFilename(
		llvm::cl::Positional,
		llvm::cl::desc("<input forth file>"),
		llvm::cl::init("-"),
		llvm::cl::value_desc("filename")),
  inputType(
		"input", llvm::cl::init(Unknown), llvm::cl::desc("Decided the kind of output desired"),
		llvm::cl::values(clEnumValN(Forth, "forth", "load the input file as a Forth file.")),
		llvm::cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file"))),
  emitAction(
		"emit", llvm::cl::desc("Select the kind of output desired"),
		llvm::cl::values(clEnumValN(DumpAST,        "ast",         "output the AST dump")),
		llvm::cl::values(clEnumValN(DumpMLIR,       "mlir",        "output the MLIR dump")),
		llvm::cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the MLIR dump after affine lowering")),
		llvm::cl::values(clEnumValN(DumpMLIRLLVM,   "mlir-llvm",   "output the MLIR dump after llvm lowering")),
		llvm::cl::values(clEnumValN(DumpLLVMIR,     "llvm",        "output the LLVM IR dump")),
		llvm::cl::values(clEnumValN(RunJIT,         "jit",         "JIT the code and run it by invoking the main function"))),
  enableOpt(
		"opt", llvm::cl::desc("Enable optimizations")),
  context(registry)
{
	// Register a bunch of command line options.
	mlir::registerAsmPrinterCLOptions();
	mlir::registerMLIRContextCLOptions();
	mlir::registerPassManagerCLOptions();
	llvm::cl::ParseCommandLineOptions(argc, argv, "forth compiler\n");


	mlir::func::registerAllExtensions(registry);
	// Load our Dialect in this MLIR Context.
	context.getOrLoadDialect<ir1::ForthDialect>();
	context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
	//context.loadDialect<mlir::LLVM::LLVMDialect>();
}

mlir::LogicalResult Main::run() {

	// *** Load input file ***

	if( inputType == forth::Main::InputType::Forth
	|| (inputType == forth::Main::InputType::Unknown && llvm::StringRef(inputFilename).ends_with(".forth"))) {
		// Handle '.forth' input to the compiler.
		if(mlir::failed(loadFileForth())) {
			return mlir::failure();
		}
	}
	else if((inputType == InputType::MLIR)
	|| (inputType == InputType::Unknown && llvm::StringRef(inputFilename).ends_with(".mlir"))) {
		// Handle '.mlir' input to the compiler.
		if(mlir::failed(loadFileMLIR())) {
			return mlir::failure();
		}
	}
	else {
		llvm::errs() << "No input file loaded\n";
		return mlir::failure();
	}


	// *** Compilation ***

	// Check to see what granularity of MLIR we are compiling to.
	bool isLoweringToAffine = emitAction >= ActionType::DumpMLIRAffine;
	bool isLoweringToLLVM = emitAction >= ActionType::DumpMLIRLLVM;
	// process lowering and optimization
	if(mlir::failed(process(enableOpt, isLoweringToAffine, isLoweringToLLVM))) {
		return mlir::failure();
	}


	// *** Dump or Execute ***

	switch(emitAction) {
	case None:
		break;
	case DumpAST:
		return dumpAST();
	// If we aren't exporting to non-mlir, then we are done.
	case DumpMLIR:
	case DumpMLIRAffine:
	case DumpMLIRLLVM:
		return dumpMLIR();
	// Check to see if we are compiling to LLVM IR.
	case DumpLLVMIR:
		return dumpLLVMIR();
	// Otherwise, we must be running the jit.
	case RunJIT:
		return runJit();
	default:
		llvm::errs() << "Unkown action specified when used -emit=<action>\n";
		return mlir::failure();
	}


	return mlir::success();
}

mlir::LogicalResult Main::process(bool enableOpt, bool loweringToAffine, bool loweringToLLVM) {
	mlir::PassManager pm(module.get()->getName());

	// Apply any generic pass manager command line options and run the pipeline.
	if(mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
		return mlir::failure();
	}

	if (enableOpt || loweringToAffine) {
		// Inline all functions into main and then delete them.
		pm.addPass(mlir::createInlinerPass());

		// Now that there is only one function, we can infer the shapes of each of the operations.
		mlir::OpPassManager &optPM = pm.nest<forth::ir1::FuncOp>();
		optPM.addPass(mlir::createCanonicalizerPass());
#if 0
		optPM.addPass(ir1::createShapeInferencePass());
#endif
		optPM.addPass(mlir::createCanonicalizerPass());
		optPM.addPass(mlir::createCSEPass());
	}

	if (loweringToAffine) {
		// Partially lower the forth dialect.
#if 0
		pm.addPass(ir1::createLowerToAffinePass());
#endif

		// Add a few cleanups post lowering.
		mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
		optPM.addPass(mlir::createCanonicalizerPass());
		optPM.addPass(mlir::createCSEPass());

		// Add optimizations if enabled.
		if (enableOpt) {
			optPM.addPass(mlir::affine::createLoopFusionPass());
			optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
		}
	}

	if (loweringToLLVM) {
		// Finish lowering forth IR to the LLVM dialect.
#if 0
		pm.addPass(ir1::createLowerToLLVMPass());
#endif
		// This is necessary to have line tables emitted and basic
		// debugger working. In the future we will add proper debug information
		// emission directly from our frontend.
		pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
	}


	if(mlir::failed(pm.run(*module))) {
		return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult Main::dumpAST() {
	if(!forthAST) {
		llvm::errs() << "Can't dump a Forth AST when there is no forth input file loaded.\n";
		return mlir::failure();
	}

	llvm::outs() << "Result:\n";
	for(const auto& ptr : *forthAST) {
		ptr->dump(0);
	}
	return mlir::success();
}

mlir::LogicalResult Main::dumpMLIR() {
	if(!module) {
		llvm::errs() << "Internal error: Module not initialized.\n";
		return mlir::failure();
	}
	module->dump();
	return mlir::success();
}

mlir::LogicalResult Main::dumpLLVMIR() {
	// Register the translation to LLVM IR with the MLIR context.
	mlir::registerBuiltinDialectTranslation(*module->getContext());
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// Convert the module to LLVM IR in a new LLVM IR context.
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return mlir::failure();
	}

	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Create target machine and configure the LLVM Module
	auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
	if (!tmBuilderOrError) {
		llvm::errs() << "Could not create JITTargetMachineBuilder\n";
		return mlir::failure();
	}

	auto tmOrError = tmBuilderOrError->createTargetMachine();
	if (!tmOrError) {
		llvm::errs() << "Could not create TargetMachine\n";
		return mlir::failure();
	}
	mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tmOrError.get().get());

	/// Optionally run an optimization pipeline over the llvm module.
	auto optPipeline = mlir::makeOptimizingTransformer(
	/*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
	/*targetMachine=*/nullptr);
	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		return mlir::failure();
	}

	llvm::outs() << *llvmModule << "\n";

	return mlir::success();
}

mlir::LogicalResult Main::runJit() {
	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Register the translation from MLIR to LLVM IR, which must happen before we
	// can JIT-compile.
	mlir::registerBuiltinDialectTranslation(*module->getContext());
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// An optimization pipeline to use within the execution engine.
	auto optPipeline = mlir::makeOptimizingTransformer(/*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

	// Create an MLIR execution engine. The execution engine eagerly JIT-compiles
	// the module.
	mlir::ExecutionEngineOptions engineOptions;
	engineOptions.transformer = optPipeline;
	auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
	assert(maybeEngine && "failed to construct an execution engine");
	auto &engine = maybeEngine.get();

	// Invoke the JIT-compiled function.
	auto invocationResult = engine->invokePacked("main");
	if (invocationResult) {
		llvm::errs() << "JIT invocation failed\n";
		return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult Main::loadFileForth() {
	llvm::StringRef filename(inputFilename);
	std::ifstream file;

	if(filename.str() != "-") {
		file.open(filename.str());
		if(file.fail()) {
			llvm::errs() << "Could not open input file: " << strerror(errno) << "\n";
			return mlir::failure();
		}
	}

	std::istream& in = (filename.str() != "-") ? file : std::cin;
	if(in.fail()) {
		llvm::errs() << "Could not open input file: " << strerror(errno) << "\n";
		return mlir::failure();
	}

	lang::Lexer lexer(in);
	forthAST.reset(new forth::lang::Block);
	lang::Parser parser(*forthAST, lexer);
	if(parser.parse() != 0) {
		llvm::errs() << "Parsing error\n";
		return mlir::failure();
	}

	/// Stateful helper class to create IR. Stateful means in the terms of "insertion points", where to put the next operation.
	mlir::OpBuilder builder(&context);

	/// A "module" matches a forth source file
	// We create an empty MLIR module and codegen functions one at a time and add them to the module.
	module = mlir::ModuleOp::create(builder.getUnknownLoc());

	if(mlir::failed(ir1::Generator::generate(builder, *module, *forthAST))) {
		llvm::errs() << "Internal error: Could not convert AST to MLIR module.\n";
		return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult Main::loadFileMLIR() {
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
	if (std::error_code ec = fileOrErr.getError()) {
		llvm::errs() << "Could not open input file: " << ec.message() << "\n";
		return mlir::failure();
	}

	// Parse the input mlir.
	llvm::SourceMgr sourceMgr;
	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

	module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
	if(!module) {
		llvm::errs() << "Error can't load file " << inputFilename << "\n";
		return mlir::failure();
	}

	return mlir::success();
}

} /* namespace forth */
