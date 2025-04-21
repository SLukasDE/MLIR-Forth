#include "forth/Main.hpp"
#include "forth/lang/Lexer.hpp"
#include "forth/lang/Parser.hpp"
#include "forth/lang/Test.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string_view>

namespace forth {

namespace {
void printUsage() {
	std::cout <<
			"OVERVIEW: forth compiler\n"
			"\n"
			"USAGE: forth <input forth file>\n";
}

} /* anonymous namespace */

Main::Main(int argc, char *argv[]) {
	if(argc != 2) {
		printUsage();
		return;
	}

	inputFilename = argv[1];
}

bool Main::run() {
	// *** Load input file ***
	if(loadFileForth() == false) {
		return false;
	}

	// *** Dump AST ***
	return dumpAST();
}

bool Main::dumpAST() {
	if(!forthAST) {
		std::cerr << "Can't dump a Forth AST when there is no forth input file loaded.\n";
		return false;
	}

	std::cout << "Result:\n";
	for(const auto& ptr : *forthAST) {
		ptr->dump(0);
	}

	return true;
}

bool Main::loadFileForth() {
	std::ifstream file;

	if(inputFilename.empty()) {
		return false;
	}

	if(inputFilename != "-") {
		file.open(inputFilename);
		if(file.fail()) {
			std::cerr << "Could not open input file: " << std::strerror(errno) << "\n";
			return false;
		}
	}

	std::istream& in = (inputFilename != "-") ? file : std::cin;

	lang::Lexer lexer(in);
	forthAST.reset(new forth::lang::Block);
	lang::Parser parser(*forthAST, lexer);
	if(parser.parse() != 0) {
		std::cerr << "Parsing error\n";
		return false;
	}

	return true;
}

} /* namespace forth */
