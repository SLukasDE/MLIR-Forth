#ifndef FORTH_LANG_LEXER_H_
#define FORTH_LANG_LEXER_H_

#if ! defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif

#include <istream>

namespace forth {
namespace lang {

class Driver;

class Lexer : public yyFlexLexer {
public:
	Lexer(std::istream& in);
	virtual ~Lexer() = default;

	virtual int fetchNextToken(void* const semanticType, void* location);
};

} /* namespace lang */
} /* namespace forth */

#endif /* FORTH_LANG_LEXER_H_ */
