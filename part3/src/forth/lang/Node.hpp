#ifndef FORTH_LANG_NODE_H_
#define FORTH_LANG_NODE_H_

#include "llvm/Support/Casting.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace forth {
namespace lang {


/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line = -1;                          ///< line number.
  int col = -1;                           ///< column number.
};

class Node;

using Block = std::vector<std::unique_ptr<Node>>;

class Node {
public:
	enum class NodeKind {
		comment,          // ( ... )
		number,           // 0|(-?[1-9][0-9]*)
		word,
		procedure,        // : <Name> ... ;
		variable,         // VARIABLE <Name>
		addrOfWord,       // ' WORD
		emitString,       // ." ..."
		ifElseEndif,      // IF ... [ELSE ...] ENDIF
		doLoop,           // DO ... LOOP
		beginUntil,       // BEGIN ... UNTIL
		beginWhileRepeat, // BEGIN ... WHILE ... REPEAT
		beginAgain        // BEGIN ... AGAIN
	};

	Node(NodeKind aKind, const Location& aLocation = Location())
	: location(aLocation),
	  kind(aKind)
	{ }

	virtual ~Node() = default;

	virtual void dump(std::size_t) const = 0;

	static std::string getSpaces(std::size_t indention) {
		std::string str;
		for(std::size_t i=0; i<indention; ++i) {
			str += "  ";
		}
		return str;
	}

	static void dumpBlock(const Block& block, std::size_t indention) {
		for(const auto& ptr : block) {
			ptr->dump(indention+1);
		}
	}

	std::string getKindStr() const {
		switch(kind) {
		case NodeKind::comment:
			return "[COMMENT           ]";
		case NodeKind::number:
			return "[NUMBER            ]";
		case NodeKind::word:
			return "[WORD              ]";
		case NodeKind::procedure:
			return "[PROCEDURE         ]";
		case NodeKind::variable:
			return "[VARIABLE          ]";
		case NodeKind::addrOfWord:
			return "[ADDR-OF-WORD      ]";
		case NodeKind::emitString:
			return "[EMIT-STRING       ]";
		case NodeKind::ifElseEndif:
			return "[IF-ELSE-ENDIF     ]";
		case NodeKind::doLoop:
			return "[DO-LOOP           ]";
		case NodeKind::beginUntil:
			return "[BEGIN-UNTIL       ]";
		case NodeKind::beginWhileRepeat:
			return "[BEGIN-WHILE-REPEAT]";
		case NodeKind::beginAgain:
			return "[BEGIN-AGAIN       ]";
		};
		return "unknown";
	}

	NodeKind getKind() const {
		return kind;
	}

	const Location location;

private:
	const NodeKind kind;
};

struct ASTNode {
	int number;
	std::string str;
	Node* node = nullptr;
	std::vector<Node*> block;

	Block makeBlock() {
	  Block rv;
      for(Node* node : block) {
        rv.emplace_back(std::unique_ptr<Node>(node));
      }
      return rv;
	}
};

class NodeNumber : public Node {
public:
	NodeNumber(int aNumber, const Location& aLocation = Location())
	: Node(NodeKind::number, aLocation),
	  number(aNumber)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << number << "\n";
	}

	const int number;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::number;
	}
};

class NodeWord : public Node {
public:
	NodeWord(const std::string& aWord)
	: Node(NodeKind::word),
	  word(aWord)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << word << "\n";
	}

	const std::string word;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::word;
	}
};

class NodeComment : public Node {
public:
	NodeComment(const std::string& aStr)
	: Node(NodeKind::comment),
	  str(aStr)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << str << "\n";
	}

	const std::string str;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::comment;
	}
};

class NodeProcedure : public Node {
public:
	NodeProcedure(const std::string& aName, Block&& aBlock)
	: Node(NodeKind::procedure),
	  name(aName),
	  body(std::move(aBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << ": " << name << "\n";
		dumpBlock(body, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << ";\n";
	}

	const std::string name;
	const Block body;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::procedure;
	}
};

class NodeVariable : public Node {
public:
	NodeVariable(const std::string& aName)
	: Node(NodeKind::variable),
	  name(aName)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "VARIABLE " << name << "\n";
	}

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::variable;
	}

	const std::string name;
};

class NodeAddrOfWord : public Node {
public:
	NodeAddrOfWord(const std::string& aWord)
	: Node(NodeKind::addrOfWord),
	  word(aWord)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "' " << word << "\n";
	}

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::addrOfWord;
	}

	const std::string word;
};

class NodeEmitString : public Node {
public:
	NodeEmitString(const std::string& aStr)
	: Node(NodeKind::emitString),
	  str(aStr)
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << ".\" " << str << "\"\n";
	}

	const std::string str;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::emitString;
	}
};

class NodeIfElseEndif : public Node {
public:
	NodeIfElseEndif(Block&& aIfBlock, Block&& aElseBlock)
	: Node(NodeKind::ifElseEndif),
	  ifBlock(std::move(aIfBlock)),
	  elseBlock(std::move(aElseBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "IF\n";
		dumpBlock(ifBlock, indention+1);
		std::cout << getKindStr() << " " << getSpaces(indention) << "ELSE\n";
		dumpBlock(elseBlock, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "ENDIF\n";
	}

	Block ifBlock;
	Block elseBlock;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::ifElseEndif;
	}
};

class NodeDoLoop : public Node {
public:
	NodeDoLoop(Block&& aBlock)
	: Node(NodeKind::doLoop),
	  block(std::move(aBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout << getKindStr() << " " << getSpaces(indention) << "DO\n";
		dumpBlock(block, indention+1);
		std::cout << getKindStr() << " " << getSpaces(indention) << "LOOP\n";
	}

	Block block;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::doLoop;
	}
};

class NodeBeginUntil : public Node {
public:
	NodeBeginUntil(Block&& aBlock)
	: Node(NodeKind::beginUntil),
	  block(std::move(aBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "BEGIN\n";
		dumpBlock(block, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "UNTIL\n";
	}

	Block block;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::beginUntil;
	}
};

class NodeBeginWhileRepeat : public Node {
public:
	NodeBeginWhileRepeat(Block&& aBeginBlock, Block&& aWhileBlock)
	: Node(NodeKind::beginWhileRepeat),
	  beginBlock(std::move(aBeginBlock)),
	  whileBlock(std::move(aWhileBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "BEGIN\n";
		dumpBlock(beginBlock, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "WHILE\n";
		dumpBlock(whileBlock, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "REPEAT\n";
	}

	Block beginBlock;
	Block whileBlock;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::beginWhileRepeat;
	}
};

class NodeBeginAgain : public Node {
public:
	NodeBeginAgain(Block&& aBlock)
	: Node(NodeKind::beginAgain),
	  block(std::move(aBlock))
	{ }

	void dump(std::size_t indention) const override {
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "BEGIN\n";
		dumpBlock(block, indention+1);
		std::cout
		<< getSpaces(indention) << getKindStr() << " "
		<< getSpaces(indention) << "AGAIN\n";
	}

	Block block;

	/// LLVM style RTTI
	static bool classof(const Node *node) {
		return node->getKind() == NodeKind::beginAgain;
	}
};

} /* namespace lang */
} /* namespace forth */

#endif /* FORTH_LANG_NODE_H_ */
