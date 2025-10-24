PROGRAM DEMO
INTEGER I, N
REAL X
I = 1;
N = 10;
X = (I + 3) * 2.5;
IF (I < N) THEN
  I = I + 1;
ELSE
  X = X - 1.0;
ENDIF
DO I = 1, N, 1
  X = X + I;
ENDDO
END
