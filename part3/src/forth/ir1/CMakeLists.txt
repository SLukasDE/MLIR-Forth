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
add_public_tablegen_target(MLIRForthPassesIncGen)
add_public_tablegen_target(MLIRForthRewriterIncGen)
message(STATUS "MLIR TableGen Ops DONE")

