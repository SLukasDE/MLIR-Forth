VARIABLE hello-token
: one ( n -- n' )  1 ;
one one + one - one * / ( das ist ein kommentar ) ." Emit
 string" ( das 
ist ein kommentar )

' one hello-token !
hello-token @ EXECUTE
