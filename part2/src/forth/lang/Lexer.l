%{
/* C++ string header, for string ops below */
#include <cstdlib>
#include <string>
 
/* Implementation of yyFlexScanner */ 
#include "forth/lang/Lexer.hpp"
#include "forth/lang/Parser.hpp"

namespace forth {
namespace lang {

namespace {

Parser::semantic_type* toSemanticType(void* const aSemanticType) {
	return static_cast<Parser::semantic_type*>(const_cast<void*>(aSemanticType));
}

Parser::location_type* toLocationType(void* aLocationType) {
	return static_cast<Parser::location_type*>(aLocationType);
}

} /* anonymous namespace */

Lexer::Lexer(std::istream& in)
: yyFlexLexer(in, std::cout)
{ };

} /* namespace condition */
} /* namespace batchelor */


/* msvc2010 requires that we exclude this header file. */
#define YY_NO_UNISTD_H

#undef  YY_DECL
#define YY_DECL int forth::lang::Lexer::fetchNextToken(void* const aSemanticType, void* aLocation)

/* update location on matching */
/*
#define YY_USER_ACTION \
	if(location) { \
		location->step(); \
		location->columns(yyleng); \
	}
*/

/* define yyterminate as this instead of NULL */
//#define yyterminate() return( END )

%}

%option debug
%option nodefault
%option yyclass="forth::lang::Lexer"
%option yylineno
%option noyywrap
%option c++

COMMENT_REGEX                 \(\ ([^ ]|\ +[^ )])*\ +\)
EMIT_STR_REGEX                \.\"\ (\\.|[^"\\])*\"
WORD_REGEX                    [\x21-\x7e]+
COMMENT_REGEX_OLD             \(\ (^(\ \)))*\ \)
EMIT_STR_REGEX_OLD            \."\ [\x21-\x7e]+"


%%

%{
/** Code executed at the beginning of lexer(...) **/
auto semanticType = toSemanticType(aSemanticType);
auto location = toLocationType(aLocation);
%}

{COMMENT_REGEX}			{
							std::string str(YYText());
							if(semanticType) {
								semanticType->str = str.substr(2, str.size()-4);
							}
							return Parser::token_type::COMMENT;
						}

{EMIT_STR_REGEX}		{
							std::string str(YYText());
							if(semanticType) {
								semanticType->str = str.substr(3, str.size()-4);
							}
							return Parser::token_type::EMIT_STR;
						}

[ \t\v\f]				;

[\n\r]					{
							// Update line number
							if(location) {
								location->lines();
							}
						}

:						{
							return Parser::token_type::PROCEDURE_BEGIN;
						}

;						{
							return Parser::token_type::PROCEDURE_END;
						}

'						{
							return Parser::token_type::ADDR_OF_WORD;
						}

VARIABLE				{
							return Parser::token_type::VARIABLE;
						}

IF						{
							return Parser::token_type::IF;
						}

ELSE					{
							return Parser::token_type::ELSE;
						}

ENDIF					{
							return Parser::token_type::ENDIF;
						}
						
DO						{
							return Parser::token_type::DO;
						}

LOOP					{
							return Parser::token_type::LOOP;
						}

BEGIN					{
							return Parser::token_type::BEGIN_LOOP;
						}

UNTIL					{
							return Parser::token_type::UNTIL;
						}

WHILE					{
							return Parser::token_type::WHILE;
						}

REPEAT					{
							return Parser::token_type::REPEAT;
						}

AGAIN					{
							return Parser::token_type::AGAIN;
						}

0|(-?[1-9][0-9]*)		{
							if(semanticType) {
								semanticType->number = std::atoi(YYText());
							}
                            return Parser::token_type::NUMBER;
						}

{WORD_REGEX}			{
							if(semanticType) {
								semanticType->str = YYText();
							}
							return Parser::token_type::WORD;
						}

.						{
							return yytext[0];
						}

%%


