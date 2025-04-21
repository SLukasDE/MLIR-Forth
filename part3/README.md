# Forth in MLIR representation

Let's build the binary:

```
cmake -G "Unix Makefiles" -S part3 -B part3/build
cmake --build part3/build
```

You can execute the binary with a forth-file as input. If you call the binary with "-" as input file, then it will read from stdin.
The program will print an AST if you run it with "--emit=ast".
If you run the program with "--emit=mlir", then you will get the MLIR code.
```
./part3/build/bin/mlir-forth part3/test.forth --emit=mlir
```

Output:
```
"builtin.module"() ({
  "llvm.mlir.global"() <{addr_space = 0 : i32, global_type = i32, linkage = #llvm.linkage<internal>, sym_name = "hello-token", value = 0 : i32, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "DUP"}> ({
  ^bb0(%arg17: tensor<?xi32>):
    %75 = "forth.get"(%arg17) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %76 = "forth.push"(%arg17, %75) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%76) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "DROP"}> ({
  ^bb0(%arg16: tensor<?xi32>):
    %74 = "forth.pop"(%arg16) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    "forth.return"(%74) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "OVER"}> ({
  ^bb0(%arg15: tensor<?xi32>):
    %72 = "forth.get"(%arg15) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %73 = "forth.push"(%arg15, %72) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%73) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "PICK"}> ({
  ^bb0(%arg14: tensor<?xi32>):
    %68 = "forth.get"(%arg14) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %69 = "forth.pop"(%arg14) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %70 = "forth.pick"(%69, %68) : (tensor<?xi32>, i32) -> i32
    %71 = "forth.push"(%69, %70) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%71) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "ROT"}> ({
  ^bb0(%arg13: tensor<?xi32>):
    %61 = "forth.get"(%arg13) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %62 = "forth.get"(%arg13) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %63 = "forth.get"(%arg13) <{n = 2 : ui32}> : (tensor<?xi32>) -> i32
    %64 = "forth.pop"(%arg13) <{n = 3 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %65 = "forth.push"(%64, %62) : (tensor<?xi32>, i32) -> tensor<?xi32>
    %66 = "forth.push"(%65, %63) : (tensor<?xi32>, i32) -> tensor<?xi32>
    %67 = "forth.push"(%66, %61) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%67) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "SWAP"}> ({
  ^bb0(%arg12: tensor<?xi32>):
    %56 = "forth.get"(%arg12) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %57 = "forth.get"(%arg12) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %58 = "forth.pop"(%arg12) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %59 = "forth.push"(%58, %56) : (tensor<?xi32>, i32) -> tensor<?xi32>
    %60 = "forth.push"(%59, %57) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%60) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "@"}> ({
  ^bb0(%arg11: tensor<?xi32>):
    %52 = "forth.get"(%arg11) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %53 = "forth.pop"(%arg11) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %54 = "llvm.load"(%52) <{ordering = 0 : i64}> : (i32) -> i32
    %55 = "forth.push"(%53, %54) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%55) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "!"}> ({
  ^bb0(%arg10: tensor<?xi32>):
    %49 = "forth.get"(%arg10) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %50 = "forth.get"(%arg10) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %51 = "forth.pop"(%arg10) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    "llvm.store"(%49, %50) <{ordering = 0 : i64}> : (i32, i32) -> ()
    "forth.return"(%51) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "EXECUTE"}> ({
  ^bb0(%arg9: tensor<?xi32>):
    %46 = "forth.get"(%arg9) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %47 = "forth.pop"(%arg9) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %48 = "forth.execute"(%47, %46) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%48) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "EMIT"}> ({
  ^bb0(%arg8: tensor<?xi32>):
    %44 = "forth.get"(%arg8) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    "forth.writeChar"(%44) : (i32) -> ()
    %45 = "forth.pop"(%arg8) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    "forth.return"(%45) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "."}> ({
  ^bb0(%arg7: tensor<?xi32>):
    %42 = "forth.get"(%arg7) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    "forth.writeInteger"(%42) : (i32) -> ()
    %43 = "forth.pop"(%arg7) <{n = 1 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    "forth.return"(%43) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "CR"}> ({
  ^bb0(%arg6: tensor<?xi32>):
    "forth.writeString"() <{text = @"\0A"}> : () -> ()
    "forth.return"(%arg6) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "KEY"}> ({
  ^bb0(%arg5: tensor<?xi32>):
    %40 = "forth.readChar"() : () -> i32
    %41 = "forth.push"(%arg5, %40) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%41) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "+"}> ({
  ^bb0(%arg4: tensor<?xi32>):
    %35 = "forth.get"(%arg4) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %36 = "forth.get"(%arg4) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %37 = "forth.pop"(%arg4) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %38 = "forth.add"(%35, %36) : (i32, i32) -> i32
    %39 = "forth.push"(%37, %38) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%39) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "-"}> ({
  ^bb0(%arg3: tensor<?xi32>):
    %30 = "forth.get"(%arg3) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %31 = "forth.get"(%arg3) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %32 = "forth.pop"(%arg3) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %33 = "forth.sub"(%30, %31) : (i32, i32) -> i32
    %34 = "forth.push"(%32, %33) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%34) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "*"}> ({
  ^bb0(%arg2: tensor<?xi32>):
    %25 = "forth.get"(%arg2) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %26 = "forth.get"(%arg2) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %27 = "forth.pop"(%arg2) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %28 = "forth.mul"(%25, %26) : (i32, i32) -> i32
    %29 = "forth.push"(%27, %28) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%29) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "/"}> ({
  ^bb0(%arg1: tensor<?xi32>):
    %20 = "forth.get"(%arg1) <{n = 0 : ui32}> : (tensor<?xi32>) -> i32
    %21 = "forth.get"(%arg1) <{n = 1 : ui32}> : (tensor<?xi32>) -> i32
    %22 = "forth.pop"(%arg1) <{n = 2 : ui32}> : (tensor<?xi32>) -> tensor<?xi32>
    %23 = "forth.div"(%20, %21) : (i32, i32) -> i32
    %24 = "forth.push"(%22, %23) : (tensor<?xi32>, i32) -> tensor<?xi32>
    "forth.return"(%24) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = (tensor<?xi32>) -> tensor<?xi32>, sym_name = "one"}> ({
  ^bb0(%arg0: tensor<?xi32>):
    %18 = "forth.constant"() <{value = 1 : i32}> : () -> tensor<?xi32>
    %19 = "forth.push"(%arg0, %18) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    "forth.return"(%19) : (tensor<?xi32>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "forth.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "forth.stack"() {stackSize = 65535 : ui32} : () -> tensor<?xi32>
    %1 = "forth.call"(%0) <{callee = @one}> : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "forth.call"(%1) <{callee = @one}> : (tensor<?xi32>) -> tensor<?xi32>
    %3 = "forth.call"(%2) <{callee = @"+"}> : (tensor<?xi32>) -> tensor<?xi32>
    %4 = "forth.call"(%3) <{callee = @one}> : (tensor<?xi32>) -> tensor<?xi32>
    %5 = "forth.call"(%4) <{callee = @"-"}> : (tensor<?xi32>) -> tensor<?xi32>
    %6 = "forth.call"(%5) <{callee = @one}> : (tensor<?xi32>) -> tensor<?xi32>
    %7 = "forth.call"(%6) <{callee = @"*"}> : (tensor<?xi32>) -> tensor<?xi32>
    %8 = "forth.call"(%7) <{callee = @"/"}> : (tensor<?xi32>) -> tensor<?xi32>
    "forth.writeString"() <{text = @"Emit\0A string"}> : () -> ()
    %9 = "forth.constant"() <{value = 18 : i32}> : () -> tensor<?xi32>
    %10 = "forth.push"(%8, %9) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %11 = "llvm.mlir.addressof"() <{global_name = @"hello-token"}> : () -> i32
    %12 = "forth.push"(%10, %11) : (tensor<?xi32>, i32) -> tensor<?xi32>
    %13 = "forth.call"(%12) <{callee = @"!"}> : (tensor<?xi32>) -> tensor<?xi32>
    %14 = "llvm.mlir.addressof"() <{global_name = @"hello-token"}> : () -> i32
    %15 = "forth.push"(%13, %14) : (tensor<?xi32>, i32) -> tensor<?xi32>
    %16 = "forth.call"(%15) <{callee = @"@"}> : (tensor<?xi32>) -> tensor<?xi32>
    %17 = "forth.call"(%16) <{callee = @EXECUTE}> : (tensor<?xi32>) -> tensor<?xi32>
    "forth.return"(%17) : (tensor<?xi32>) -> ()
  }) : () -> ()
}) : () -> ()
```

# Dialect

Now let's dig into the code.
