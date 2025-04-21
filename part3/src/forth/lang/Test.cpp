#include "forth/lang/Test.hpp"
#include "forth/lang/Parser.hpp"
#include "forth/lang/Lexer.hpp"

#include <iostream>
#include <sstream>

namespace forth {
namespace lang {

namespace {
class TestLexer : public Lexer
{
public:
	TestLexer()
	: Lexer(fooStream)
	{ }

	int fetchNextToken(void* const semanticType, void* location) override {
		switch(state++) {
		case 0:
			return Parser::token_type::COMMENT;
		case 1:
			return Parser::token_type::EMIT_STR;
		case 2:
			return Parser::token_type::VARIABLE;
		case 3:
			return Parser::token_type::WORD;
		case 4:
			return Parser::token_type::NUMBER;
		case 5:
			return Parser::token_type::STORE;
		default:
			break;
		}
		return 0;
	}

	int state = 0;
	static std::stringstream fooStream;
};

std::stringstream TestLexer::fooStream;
}

void Test::testLexer() {
	std::cout << "TEST Scanner" << std::endl;
	std::cout << "============" << std::endl;

	std::stringstream sstr;
	sstr << testString;

	std::cout << "Input:" << std::endl;
	std::cout << sstr.str() << std::endl;
	std::cout << "---------------------" << std::endl;

	Lexer scanner(sstr);

	while(true) {
		int token = scanner.fetchNextToken(nullptr, nullptr);

		switch(token) {
		case Parser::token_type::COMMENT:
			std::cout << "COMMENT" << std::endl;
			break;
		case Parser::token_type::EMIT_STR:
			std::cout << "EMIT_STR" << std::endl;
			break;
		case Parser::token_type::PROCEDURE_BEGIN:
			std::cout << "PROCEDURE_BEGIN" << std::endl;
			break;
		case Parser::token_type::PROCEDURE_END:
			std::cout << "PROCEDURE_END" << std::endl;
			break;
		case Parser::token_type::ADDR_OF_WORD:
			std::cout << "ADDR_OF_WORD" << std::endl;
			break;
		case Parser::token_type::STORE:
			std::cout << "STORE" << std::endl;
			break;
		case Parser::token_type::VARIABLE:
			std::cout << "VARIABLE" << std::endl;
			break;
		case Parser::token_type::IF:
			std::cout << "IF" << std::endl;
			break;
		case Parser::token_type::ELSE:
			std::cout << "ELSE" << std::endl;
			break;
		case Parser::token_type::ENDIF:
			std::cout << "ENDIF" << std::endl;
			break;
		case Parser::token_type::DO:
			std::cout << "DO" << std::endl;
			break;
		case Parser::token_type::LOOP:
			std::cout << "LOOP" << std::endl;
			break;
		case Parser::token_type::BEGIN_LOOP:
			std::cout << "BEGIN_LOOP" << std::endl;
			break;
		case Parser::token_type::UNTIL:
			std::cout << "UNTIL" << std::endl;
			break;
		case Parser::token_type::WHILE:
			std::cout << "WHILE" << std::endl;
			break;
		case Parser::token_type::REPEAT:
			std::cout << "REPEAT" << std::endl;
			break;
		case Parser::token_type::AGAIN:
			std::cout << "AGAIN" << std::endl;
			break;
		case Parser::token_type::NUMBER:
			std::cout << "NUMBER" << std::endl;
			break;
		case Parser::token_type::WORD:
			std::cout << "WORD" << std::endl;
			break;
		default:
			std::cout << "Value " << std::to_string(token) << std::endl;
			if(token == 0) {
				return;
			}
		}
	}
}

void Test::testParser() {
	std::cout << "TEST Parser" << std::endl;
	std::cout << "===========" << std::endl;

	TestLexer lexer;
	Block block;

	Parser parser(block, lexer);
	if(parser.parse() != 0) {
		std::cerr << "Error" << std::endl;
	}

	std::cout << "Result:" << std::endl;
	for(const auto& ptr : block) {
		ptr->dump(0);
	}
	std::cout << "=====================" << std::endl;
}

void Test::testScannerParser1() {
	std::cout << "TEST Scanner + Parser 1" << std::endl;
	std::cout << "=======================" << std::endl;

	std::stringstream sstr;
	sstr << testString;

	std::cout << "Input:" << std::endl;
	std::cout << sstr.str() << std::endl;
	std::cout << "---------------------" << std::endl;

	Lexer lexer(sstr);
	Block block;

	Parser parser(block, lexer);
	if(parser.parse() != 0) {
		std::cerr << "Error" << std::endl;
	}
	std::cout << "Result:" << std::endl;
	for(const auto& ptr : block) {
		ptr->dump(0);
	}
	std::cout << "=====================" << std::endl;

}


} /* namespace lang */
} /* namespace forth */
