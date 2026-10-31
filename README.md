# mini_exact_polynomial_gcd

Computes the greatest common divisor of two dense, univariate polynomials with rational coefficients, using the Euclidean algorithm over Python's `Fraction` type.

```python
from fractions import Fraction
from mini_exact_polynomial_gcd import Polynomial, gcd

# (x - 1)(x + 2) and (x - 1)(x - 3)
a = Polynomial([-2, 1, 1])
b = Polynomial([3, -4, 1])

print(gcd(a, b))  # Polynomial([Fraction(-1, 1), Fraction(1, 1)])  i.e. x - 1
```

The library exists because exact polynomial GCD is the kind of thing you keep rewriting by hand: `Fraction` already gives you a field, and the Euclidean algorithm is a dozen lines, but getting the degree conventions, the zero polynomial, and the monic normalisation right is fiddly. This packages that up with no dependencies beyond the standard library.

The trade-off is density: polynomials are stored as a flat list of coefficients, lowest degree first. This is simple and fast enough for modest degrees, but it will not scale to inputs with thousands of terms where a sparse representation would win.

The result is always normalised to be monic (leading coefficient 1), so `gcd(2x+4, 3x+6)` returns `x + 2`, not `2x + 4`. The zero polynomial is represented by an empty coefficient list and has degree `-1`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

