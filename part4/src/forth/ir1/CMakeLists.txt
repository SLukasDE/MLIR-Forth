message(STATUS "MLIR TableGen Dialect ...")
set(LLVM_TARGET_DEFINITIONS ForthDialect.td)
mlir_tablegen(ForthDialect.h.inc -gen-dialect-decls -dialect=forth)
mlir_tablegen(ForthDialect.cpp.inc -gen-dialect-defs -dialect=forth)
add_public_tablegen_target(MLIRForthDialectIncGen)
message(STATUS "MLIR TableGen Dialect DONE")


message(STATUS "MLIR TableGen Ops ...")
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIROpsIncGen)
message(STATUS "MLIR TableGen Ops DONE")


message(STATUS "MLIR TableGen Passes ...")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRForthPassesIncGen)
message(STATUS "MLIR TableGen Passes DONE")


message(STATUS "MLIR TableGen Rewriter...")
set(LLVM_TARGET_DEFINITIONS Rewriter.td)
mlir_tablegen(Rewriter.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRForthRewriterIncGen)
message(STATUS "MLIR TableGen Rewriter DONE")

