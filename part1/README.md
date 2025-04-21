# The Forth Language

Forth is a very simple language, but allowes you to write low-level code, like device drivers. This is done by Open Firmware (IEEE-1275).
The main concept of Forth consist of "words" and the stack. The stack holds integer values. You can push and pop values on the stack. The next value to pop from the stack is called the "TOS" (Top Of Stack).
"Words" are the operations you can execute. You can think about that a "word" is something as a procedure. If the "word" expects arguments, then you have to put them on the stack before you can execute the word.
The "word" can manipulate the stack, so you can receive a result on the stack.

There are only very few built-in words:

| Word     | Description |
| -------- | ----------- |
| <NUMBER> | Every number is a word. It will put the number on the stack. |
| DUP      | Duplicated the TOS. |
| SWAP     | Exchanges the TOS and TOS-1. |
| ...      | ... |
| +        | Adds the elements at TOS and TOS-1, removes them and put the sum of both elements back on the stack. |
| -        | ... |
| *        | ... |
| /        | ... |
| ...      | ... |
| !        | STORE the value of TOS to the address at TOS-1 |
| @        | LOAD the value at address TOS and put it on the TOS |
| ...      | ... |
| KEY      | read char from keyboard |
| EMIT     | write TOS as char to stdout |
| .        | write TOS as number to stdout |
| CR       | write Carriage Return/Line Feed to stdout |
| ." abc"  | write 'abc' to stdout |

See [test.forth](test.forth) for more built-in words.

You can define own "words" with:
```
: <MyWord>
...
;
```
E.g.
```
: DUP2 DUP DUP ;
```
Now you can use the word "DUP2" as well, that duplicates the TOS twice.

There are also conditions, like "IF...ELSE...ENDIF", loops and words for IO operations.

Comments start with "( " and end with " )".

