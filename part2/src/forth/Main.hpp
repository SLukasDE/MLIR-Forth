#ifndef FORTH_MAIN_H_
#define FORTH_MAIN_H_

#include "forth/lang/Node.hpp"

#include <memory>
#include <string>

namespace forth {

class Main {
public:
	Main(int argc, char *argv[]);

	bool run();

private:
	std::string inputFilename;

	std::unique_ptr<forth::lang::Block> forthAST;

	bool dumpAST();
	bool loadFileForth();
};

} /* namespace forth */

#endif /* FORTH_MAIN_H_ */
