include "ForthDialect.td"

//===----------------------------------------------------------------------===//
// ForthOp
//===----------------------------------------------------------------------===//

class ForthOp<string mnemonic, list<Trait> traits = []> : Op<ForthDialect, mnemonic, traits>
{
  let summary = "Operation Class";
}
