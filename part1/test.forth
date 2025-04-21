DUP     ( n1 n2        -> n1 n2 n2      dupliziert das oberste Stack-Element )
SWAP    ( n1 n2 n3     -> n1 n3 n2      vertauscht die obersten beiden Stack-Elemente )
ROT     ( n1 n2 n3     -> n2 n3 n1      holt das dritte Stack-Element nach oben )
OVER    ( n1 n2 n3     -> n1 n2 n3 n2   kopiert das zweite Stack-Element )
PICK    ( n1 n2 n3 2   -> n1 n2 n3 n1   kopiert das angegebene (hier: 2 entspr. dritte) Stack-Element )
DROP    ( n1 n2 n3 n4  -> n1 n2 n3      entfernt das oberste Stack-Element )

( +	Addition )
( -	Subtraktion )
( *	Multiplikation )
( /	Division )
( MOD	Divisionsrest )
( /MOD	Rest und Ergebnis bei Division )
( MAX	Maximalwert )
( MIN	Minimalwert )
( ABS	Absolutwert )
( MINUS	Vorzeichenwechsel )
( AND	Logisches UND )
( OR	Logisches ODER )
( XOR	Logisches EXCLUSIV ODER )
( <	Test ob kleiner )
( >	Test ob größer )
( =	Test ob gleich )
( 0<	Test ob negativ )
( 0=	Test ob Null )

( ---- Conditions and Loops ---- )

( IF... ENDIF                - Die IF-Abfrage testet, ob der TOS (Top of Stack, also das oberste Element des Stacks) den Wert Null (= False) oder einen Wert ungleich Null (= True) hat.
                               Ist das Ergebnis "True", dann werden die Anweisungen zwischen IF und ENDIF ausgeführt, andernfalls wird das Programm nach ENDIF fortgesetzt.
                               Einige ältere Forth-Versionen benutzen das Schlüsselwort THEN statt ENDIF. )

( IF ... ELSE ... ENDIF      - Arbeitet ähnlich der einfachen IF-Abfrage, führt aber im Falle TOS=0 die Anweisungen zwischen ELSE und ENDIF aus, sonst die Anweisungen zwischen IF und ELSE. )

( DO ... LOOP                - Entspricht der FOR-NEXT-Anweisung in Basic. Das Wort DO erwartet zwei Parameter auf dem Stack, nämlich die Anfangs- und die Endzahl der Schleifendurchläufe.
                               Mit LOOP wird die Schleife beendet. Falls das Wort + LOOP statt LOOP verwendet wird, erhöht sich der Schleifenindex nicht um 1, sondern um den Wert des TOS.
                               Das Wort "1" wird innerhalb von Schleifen dazu verwendet, den Schleifenzähler auf den Stack zu kopieren. )

( BEGIN ... UNTIL            - Diese Schleife wird solange durchlaufen, bis UNTIL einen Wert ungleich Null (also "True") auf dem Stack vorfindet. Entspricht REPEAT ... UNTIL in Pascal. )

( BEGIN ... WHILE ... REPEAT - Der Programmteil zwischen BEGIN und REPEAT wird solange durchlaufen, bis das Wort WHILE die Bedingung TOS=0 feststellt.
                               Danach wird die Schleife unmittelbar hinter WHILE verlassen. )

( BEGIN ... AGAIN            - Dient zum Programmieren unendlicher Schleifen und testet keinerlei Bedingungen. )

( ---- Read / Write from stdio ---- )
KEY       ( read char from keyboard )
EMIT      ( write TOS as char )
.         ( write TOS as number )
CR        ( write Carriage Return/Line Feed )
." Hello" ( write 'Hello' )

: floor5 ( n -- n' )
  DUP
  6 <
  IF
    DROP
    5
  ELSE
    1
    -
  ENDIF
;

: hello ( comment ... )
  ." Hello"
;

VARIABLE hello-token

( ---- Address, Load, Store, Execute at address ---- )
' hello               ( gets the address of the word 'hello' and puts it on the stack. )
!                     ( STORE the value of TOS to the address at TOS-1 )
@                     ( LOAD the value at address TOS and put it on the TOS )
EXECUTE               ( Gets an address from the stack and runs whatever word is found at that address )

( Example: )
' hello hello-token ! ( store the address of the word 'hello' into the variable 'hello-token')
hello-token @ EXECUTE ( Execute the word at address, stored in the variable 'hello-token' that is the word 'hello'. )
( > hello )

