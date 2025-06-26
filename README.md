# Warning: This tutorial is a work in progress and not even a draft so far!

# MLIR-Forth-Tutorial

If you want to learn LLVM in detail first of all you will find a very good, detailed online documentation on the project web site. There are also many bookes available, like
- LLVM Essentials - By Suyog Sarda, Mayur Pandey
- Learn LLVM 17 - By Kai Nacke, Amy Kwan
- Getting Started with LLVM Core Libraries - By Bruno Cardoso Lopes, Rafael Auler
- LLVM Cookbook - By Mayur Pandey, Suyog Sarda

If you go further to Clang, there are also two additional books available that I would recommend to read:
- LLVM Techniques - By Min-Yih Hsu
- Clang Compiler Frontend - By Ivan Murashko

When I started to lear using the LLVM compiler framework, I was looking for a small project that I can develop by myown and that will use LLVM framework.
I was developing Open Firmware for the CellBE CPU when I worked at the IBM lab. One part of Open Firmware is a forth prompt.
So, it was obvious for my to implement my own Forth Compiler with LLVM.

Now I moved forward to MLIR, but there is not so much literature available beside of the website of LLVM/MLIR itself.
Anyways, I was able to reimplement the LLVM based Forth Compiler to an MLIR based Forth Compiler. And first I did many things wrong.

Because there is less documentation available (especially for beginners) than for LLVM, I want you to show how I implemented a Forth Compiler.
Most tutorials are going straight forward to a working implementation. Instead I will take you to journey, to see as well some anti patterns and we will fix them.
Thus, we have to step a little bit backward after every part to fix the problem and then we will walk forward to the next milestone.

Now, you can just lean back, take a cup of coffee and read the tutorial. You don't need to write code by your own if you just want to learn a little bit about MLIR in theory.
By my own experience, I would recommend to checkout the code, compile it, read it in your own editor and play a little bit around. In both cases you should be able to read and write (modern) C++ code.
I use Flex and Bison to implement the Forth parser. Thus, it would be helpful, to have some knwoledge about Flex (or Lex), Bison (or YACC), Regex and ENBF gramar.
If not, then you can skip part 2 and take the generation of the AST as a black box. That's absolutley fine because this tutorial is about MLIR and not about parsers.
For those, who used Flex and Bison one in the past, it might be a good source to see how to embed these tools into you own CMake project.

## Prerequisites

I used Linux (Ubuntu 22.04) and I didn't tested the code on other platforms so far. Thus, if you use Windows, I recommend you to build this code with WLS2.
MacOS users should be able to use this code as it is. I built some parts on MacOS (Sequoia), but not the whole tutorial so far. I don't expect any problems on MacOS.

### For Linux and WSL Users with Debian or Ubuntu:

* Install a compiler/linker ("clang" or "gcc")
* Install CMake
* Install Ninja (the modern way) or GNU Make (old but gold)
* Install flex and bison

E.g. Ninja and Clang:
```
sudo apt install cmake ninja-build clang lld flex bison
```

Or GCC and GNU Make:
```
sudo apt install cmake build-essential flex bison
```

### Get the source code

Use "git" to get the source code :
```
git clone https://github.com/SLukas_DE/MLIR-Forth
cd MLIR-Forth
git submodule update --init
```

### Build LLVM/MLIR from source code:

```
cmake -G "Unix Makefiles" \
   -S third_party/llvm-project/llvm \
   -B third_party/llvm-project-build \
   -DCMAKE_INSTALL_PREFIX=$(pwd)/third_party/llvm-installed/ \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF
cmake --build third_party/llvm-project-build/
cmake --build third_party/llvm-project-build/ --target install
cmake --build third_party/llvm-project-build/ --target check-mlir
```

You can use "Ninja" instead of "Unix Makefiles":

```
cmake -G "Ninja" \
   -S third_party/llvm-project/llvm \
   ...
```

### Build the Forth compiler

This tutorial is divided into 5 parts, while part 1 does not contain any source code. Thus, the first code to build you will find it in part 2.
Every part is build independet from the other parts. For example, to build part 2 you cn do it like this:

```
cmake -G "Unix Makefiles" -S part2 -B part2/build
cmake --build part2/build
```

Again, you can use "Ninja" instead of "Unix Makefiles".

You will find the executable here:
```
./part2/bin/mlir-forth
```

## Content

Part 1: [Intgroduction to the Forth Language](part1/README.md)

Part 2: [Parsing the source language and generating the AST](part2/README.md)

Part 3: [Writing our own dialect](part3/README.md)

Part 4: [Compilation and Optimization](part4/README.md)

Part 5: tbd
