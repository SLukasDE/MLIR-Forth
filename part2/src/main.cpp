#include "forth/Main.hpp"

int main(int argc, char *argv[]) {
	return !forth::Main(argc, argv).run() ? -1 : 0;
}
