add_subdirectory(forth/ir1)

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Add executable and link it with the parser library
#add_llvm_executable(forth ${SRC}  DEPENDS MLIRForthDialectIncGen)
add_llvm_executable(mlir-forth ${SRC} ${CMAKE_BINARY_DIR}/src/forth/lang/Lexer.cpp ${CMAKE_BINARY_DIR}/src/forth/lang/Parser.cpp
  DEPENDS MLIRForthDialectIncGen
          MLIROpsIncGen
          MLIRForthPassesIncGen
          MLIRForthRewriterIncGen)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/forth/lang)

if(FLEX_FOUND)
  message(STATUS "FLEX found")
  add_custom_command(
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/forth/lang/Lexer.l
    COMMAND ${FLEX_EXECUTABLE}
    ARGS --outfile=${CMAKE_BINARY_DIR}/src/forth/lang/Lexer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/forth/lang/Lexer.l
    COMMENT "Generating Lexer.cpp"
    OUTPUT ${CMAKE_BINARY_DIR}/src/forth/lang/Lexer.cpp
  )
else(FLEX_FOUND)
  message(ERROR "FLEX NOT found")
endif(FLEX_FOUND)

if(BISON_FOUND)
  add_custom_command(
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/forth/lang/Parser.yy
    COMMAND ${BISON_EXECUTABLE}
    ARGS -L C++ -d -v ${CMAKE_CURRENT_SOURCE_DIR}/forth/lang/Parser.yy
    COMMENT "Generating Parser.cpp"
    OUTPUT ${CMAKE_BINARY_DIR}/src/forth/lang/Parser.cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/forth/lang/
  )
else(BISON_FOUND)
  message(ERROR "BISON NOT found")
endif(BISON_FOUND)

target_link_libraries(mlir-forth PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
    )
