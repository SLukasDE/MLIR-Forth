file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add executable and link it with the parser library
add_executable(mlir-forth ${SRC} ${CMAKE_BINARY_DIR}/src/forth/lang/Lexer.cpp ${CMAKE_BINARY_DIR}/src/forth/lang/Parser.cpp)

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

target_link_libraries(mlir-forth)
