%skeleton "lalr1.cc"
%require  "3.2"
%defines "Parser.hpp"
%output "Parser.cpp"
%define api.namespace {forth::lang}
%define api.parser.class {Parser}

// This code is copied at the beginning of the generated file Parser.hpp
%code requires{

#include "forth/lang/Node.hpp"

namespace forth {
namespace lang {

class Lexer;

} /* namespace lang */
} /* namespace forth */

}

%parse-param { Block& block }
%parse-param { Lexer& lexer }

// This code is copied at the beginning of the generated file Parser.cpp
%code{
#include <forth/lang/Lexer.hpp>
#include <string>

namespace forth {
namespace lang {

} /* namespace lang */
} /* namespace forth */
   
// Make the parser calling ...->Lexer::fetchNextToken(...) instead of ::yylex(...) 
#undef yylex
#define yylex lexer.fetchNextToken

}

//%define api.value.type variant
//%define api.value.type union
%define api.value.type {ASTNode}
%define parse.assert

%locations

%start startToken

//%token <char>               CHAR

//%token <std::string>      COMMENT         256
//%token <std::string>      EMIT_STR        257
%token                      COMMENT         256
%token                      EMIT_STR        257
%token                      PROCEDURE_BEGIN 258
%token                      PROCEDURE_END   259
%token                      ADDR_OF_WORD    260
%token                      STORE           261
%token                      VARIABLE        262
%token                      IF              263
%token                      ELSE            264
%token                      ENDIF           265
%token                      DO              266
%token                      LOOP            267
%token                      BEGIN_LOOP      268
%token                      UNTIL           269
%token                      WHILE           270
%token                      REPEAT          271
%token                      AGAIN           272
%token                      NUMBER          273
%token                      WORD            274
//%token <int>              NUMBER          273
//%token <std::string>      WORD            274

%type                            globalNode
%type                            globalNodeList
%type                            node
%type                            nodeList
//%type  <std::unique_ptr<Node>> globalNode
//%type  <Block>                 globalNodeList
//%type  <std::unique_ptr<Node>> node
//%type  <Block>                 nodeList

%%

startToken
  : globalNodeList {
      block = $1.makeBlock();
    }
  ;
  
globalNodeList
  : /* allow empty input */ { }
  | globalNodeList globalNode {
      $$.block = std::move($1.block);
      $$.block.emplace_back($2.node);
    }
  ;

globalNode
  : node {
      $$.node = $1.node;
    }
  | VARIABLE WORD {
      $$.node = new NodeVariable($2.str);
    }
  | PROCEDURE_BEGIN WORD nodeList PROCEDURE_END {
      $$.node = new NodeProcedure($2.str, $3.makeBlock());
    }
  ;

nodeList
  : /* allow empty input */ { }
  | nodeList node {
      $$.block = std::move($1.block);
      $$.block.emplace_back($2.node);
    }
  ;

node
  : EMIT_STR {
      $$.node = new NodeEmitString($1.str);
    }
  | ADDR_OF_WORD WORD {
      $$.node = new NodeAddrOfWord($2.str);
    }
  | NUMBER {
      $$.node = new NodeNumber($1.number);
    }
  | WORD {
      $$.node = new NodeWord($1.str);
    }
  | IF nodeList ENDIF {
      $$.node = new NodeIfElseEndif($2.makeBlock(), Block());
    }
  | IF nodeList ELSE nodeList ENDIF {
      $$.node = new NodeIfElseEndif($2.makeBlock(), $4.makeBlock());
    }
  | DO nodeList LOOP {
      $$.node = new NodeDoLoop($2.makeBlock());
    }
  | BEGIN_LOOP nodeList UNTIL {
      $$.node = new NodeBeginUntil($2.makeBlock());
    }
  | BEGIN_LOOP nodeList WHILE nodeList REPEAT {
      $$.node = new NodeBeginWhileRepeat($2.makeBlock(), $4.makeBlock());
    }
  | BEGIN_LOOP nodeList AGAIN {
      $$.node = new NodeBeginAgain($2.makeBlock());
    }
  | COMMENT {
      $$.node = new NodeComment($1.str);
    }
  ;

%%


void forth::lang::Parser::error(const location_type& l, const std::string& err_message) {
	std::cerr << "Error: " << err_message << " at " << l;
}
