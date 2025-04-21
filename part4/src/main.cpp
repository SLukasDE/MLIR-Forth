#include "forth/Main.hpp"

int main(int argc, char *argv[]) {
	return mlir::failed(forth::Main(argc, argv).run()) ? -1 : 0;
}
