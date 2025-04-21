#ifndef FORTH_LANG_TEST_H_
#define FORTH_LANG_TEST_H_

#include <string>

namespace forth {
namespace lang {

class Test {
public:
	Test() = default;

	void testLexer();
	void testParser();
	void testScannerParser1();

	static constexpr char testString[] = "VARIABLE foo word @ ' word w ( wurst ) .\" emitstr1\" .\" emit str 2\" : PROC test ;";
//	static constexpr char testString[] = "A B C VARIABLE x x @ .\" bla\" ( bla )";

};

} /* namespace lang */
} /* namespace forth */

#endif /* FORTH_LANG_TEST_H_ */
