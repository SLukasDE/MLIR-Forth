# Forth Parser

Let's build the parser:

```
cmake -G "Unix Makefiles" -S part2 -B part2/build
cmake --build part2/build
```

You can execute the binary with a forth-file as input. If you call the binary with "-" as input file, then it will read from stdin.
The parser will print an AST, that is very flat because of the nature of the forth language.
```
./part2/build/bin/mlir-forth part2/test.forth
```

Output:
```
Result:
[WORD              ] DUP
[COMMENT           ] n1 n2        -> n1 n2 n2      dupliziert das oberste Stack-Element
[WORD              ] SWAP
[COMMENT           ] n1 n2 n3     -> n1 n3 n2      vertauscht die obersten beiden Stack-Elemente
[WORD              ] ROT
[COMMENT           ] n1 n2 n3     -> n2 n3 n1      holt das dritte Stack-Element nach oben
[WORD              ] OVER
[COMMENT           ] n1 n2 n3     -> n1 n2 n3 n2   kopiert das zweite Stack-Element
[WORD              ] PICK
[COMMENT           ] n1 n2 n3 2   -> n1 n2 n3 n1   kopiert das angegebene (hier: 2 entspr. dritte) Stack-Element
[WORD              ] DROP
[COMMENT           ] n1 n2 n3 n4  -> n1 n2 n3      entfernt das oberste Stack-Element
[COMMENT           ] +  Addition
[COMMENT           ] -  Subtraktion
[COMMENT           ] *  Multiplikation
[COMMENT           ] /  Division
[COMMENT           ] MOD        Divisionsrest
[COMMENT           ] /MOD       Rest und Ergebnis bei Division
[COMMENT           ] MAX        Maximalwert
[COMMENT           ] MIN        Minimalwert
[COMMENT           ] ABS        Absolutwert
[COMMENT           ] MINUS      Vorzeichenwechsel
[COMMENT           ] AND        Logisches UND
[COMMENT           ] OR Logisches ODER
[COMMENT           ] XOR        Logisches EXCLUSIV ODER
[COMMENT           ] <  Test ob kleiner
[COMMENT           ] >  Test ob größer
[COMMENT           ] =  Test ob gleich
[COMMENT           ] 0< Test ob negativ
[COMMENT           ] 0= Test ob Null
[COMMENT           ] ---- Conditions and Loops ----
[COMMENT           ] IF... ENDIF                - Die IF-Abfrage testet, ob der TOS (Top of Stack, also das oberste Element des Stacks) den Wert Null (= False) oder einen Wert ungleich Null (= True) hat.
                               Ist das Ergebnis "True", dann werden die Anweisungen zwischen IF und ENDIF ausgeführt, andernfalls wird das Programm nach ENDIF fortgesetzt.
                               Einige ältere Forth-Versionen benutzen das Schlüsselwort THEN statt ENDIF.
[COMMENT           ] IF ... ELSE ... ENDIF      - Arbeitet ähnlich der einfachen IF-Abfrage, führt aber im Falle TOS=0 die Anweisungen zwischen ELSE und ENDIF aus, sonst die Anweisungen zwischen IF und ELSE.
[COMMENT           ] DO ... LOOP                - Entspricht der FOR-NEXT-Anweisung in Basic. Das Wort DO erwartet zwei Parameter auf dem Stack, nämlich die Anfangs- und die Endzahl der Schleifendurchläufe.
                               Mit LOOP wird die Schleife beendet. Falls das Wort + LOOP statt LOOP verwendet wird, erhöht sich der Schleifenindex nicht um 1, sondern um den Wert des TOS.
                               Das Wort "1" wird innerhalb von Schleifen dazu verwendet, den Schleifenzähler auf den Stack zu kopieren.
[COMMENT           ] BEGIN ... UNTIL            - Diese Schleife wird solange durchlaufen, bis UNTIL einen Wert ungleich Null (also "True") auf dem Stack vorfindet. Entspricht REPEAT ... UNTIL in Pascal.
[COMMENT           ] BEGIN ... WHILE ... REPEAT - Der Programmteil zwischen BEGIN und REPEAT wird solange durchlaufen, bis das Wort WHILE die Bedingung TOS=0 feststellt.
                               Danach wird die Schleife unmittelbar hinter WHILE verlassen.
[COMMENT           ] BEGIN ... AGAIN            - Dient zum Programmieren unendlicher Schleifen und testet keinerlei Bedingungen.
[COMMENT           ] ---- Read / Write from stdio ----
[WORD              ] KEY
[COMMENT           ] read char from keyboard
[WORD              ] EMIT
[COMMENT           ] write TOS as char
[WORD              ] .
[COMMENT           ] write TOS as number
[WORD              ] CR
[COMMENT           ] write Carriage Return/Line Feed
[EMIT-STRING       ] ." Hello"
[COMMENT           ] write 'Hello'
[PROCEDURE         ] : floor5
    [COMMENT           ]     n -- n'
    [WORD              ]     DUP
    [NUMBER            ]     6
    [WORD              ]     <
    [IF-ELSE-ENDIF     ]     IF
        [WORD              ]         DROP
        [NUMBER            ]         5
[IF-ELSE-ENDIF     ]     ELSE
        [NUMBER            ]         1
        [WORD              ]         -
    [IF-ELSE-ENDIF     ]     ENDIF
[PROCEDURE         ] ;
[PROCEDURE         ] : hello
    [COMMENT           ]     comment ...
    [EMIT-STRING       ]     ." Hello"
[PROCEDURE         ] ;
[VARIABLE          ] VARIABLE hello-token
[COMMENT           ] ---- Address, Load, Store, Execute at address ----
[ADDR-OF-WORD      ] ' hello
[COMMENT           ] gets the address of the word 'hello' and puts it on the stack.
[WORD              ] !
[COMMENT           ] STORE the value of TOS to the address at TOS-1
[WORD              ] @
[COMMENT           ] LOAD the value at address TOS and put it on the TOS
[WORD              ] EXECUTE
[COMMENT           ] Gets an address from the stack and runs whatever word is found at that address
[COMMENT           ] Example:
[ADDR-OF-WORD      ] ' hello
[WORD              ] hello-token
[WORD              ] !
[COMMENT           ] store the address of the word 'hello' into the variable 'hello-token')
hello-token @ EXECUTE ( Execute the word at address, stored in the variable 'hello-token' that is the word 'hello'.
[COMMENT           ] > hello
```
