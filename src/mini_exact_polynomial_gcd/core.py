"""Exact polynomial GCD over the rational numbers.

We store a polynomial as a list of :class:`~fractions.Fraction` coefficients
in ascending order of degree. We deliberately keep the data model tiny:
sparse representations make the Euclidean algorithm fiddly (zero-padding,
bookkeeping of exponents) and are unnecessary for inputs of modest degree.
"""

from __future__ import annotations

from fractions import Fraction

__all__ = ["Polynomial", "gcd", "InvalidPolynomialError"]


class InvalidPolynomialError(ValueError):
    """Raised when coefficients are not a sequence of :class:`~fractions.Fraction` values."""


def _as_fraction_list(coefficients) -> list[Fraction]:
    """Coerce an iterable of numbers into a list of :class:`~fractions.Fraction`.

    Accepts plain ints and floats as well as Fractions, but rejects any input
    that is not a real number. The constructor is forgiving on purpose so
    that callers can pass ``[1, -2, 3]`` rather than wrapping every integer.
    """
    result: list[Fraction] = []
    for i, c in enumerate(coefficients):
        # ``bool`` is a subclass of ``int`` in Python; we accept it since the
        # arithmetic is well defined, but convert to Fraction explicitly so
        # downstream comparisons against Fraction behave predictably.
        if isinstance(c, bool):
            result.append(Fraction(int(c)))
        elif isinstance(c, Fraction):
            result.append(c)
        elif isinstance(c, int):
            result.append(Fraction(c))
        elif isinstance(c, float):
            # Convert via the string form so that 0.5 becomes exactly 1/2
            # rather than the Fraction approximation of the binary float.
            # This is the recommended idiom in the :mod:`fractions` docs.
            result.append(Fraction(str(c)))
        else:
            raise InvalidPolynomialError(
                f"coefficient at index {i} is not a real number: {c!r}"
            )
    return result


class Polynomial:
    """A dense univariate polynomial with exact rational coefficients.

    Coefficients are stored in a list, lowest degree first. The canonical
    form always ends in a nonzero coefficient (unless the polynomial is the
    zero polynomial), so two algebraically-equal polynomials compare equal.
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coefficients=None) -> None:
        if coefficients is None:
            self._coeffs: list[Fraction] = []
        else:
            self._coeffs = _as_fraction_list(coefficients)
        # Strip trailing zeros so that degree comparisons are reliable.
        # The zero polynomial is represented as an empty list.
        self._normalize()

    def _normalize(self) -> None:
        while self._coeffs and self._coeffs[-1] == 0:
            self._coeffs.pop()

    @property
    def coefficients(self) -> list[Fraction]:
        """Return a copy of the coefficient list so callers cannot mutate us."""
        return list(self._coeffs)

    @property
    def degree(self) -> int:
        """The degree, or ``-1`` for the zero polynomial.

        We use the ``-1`` convention so that ``degree(a) < degree(b)`` works
        even when ``a`` is zero; this keeps the Euclidean algorithm's degree
        guard simple.
        """
        return len(self._coeffs) - 1

    @property
    def is_zero(self) -> bool:
        return not self._coeffs

    @classmethod
    def constant(cls, value) -> "Polynomial":
        """Build a degree-zero polynomial from a single number."""
        return cls([value])

    @classmethod
    def zero(cls) -> "Polynomial":
        """The zero polynomial."""
        return cls([])

    def leading_coefficient(self) -> Fraction:
        """The highest-degree nonzero coefficient.

        Returns ``Fraction(0)`` for the zero polynomial, which avoids a
        special case in callers that divide by this value (they will instead
        hit the zero-polynomial branch on a degree check first).
        """
        if not self._coeffs:
            return Fraction(0)
        return self._coeffs[-1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self._coeffs == other._coeffs

    def __hash__(self) -> int:
        return hash(tuple(self._coeffs))

    def __repr__(self) -> str:
        return f"Polynomial({self._coeffs!r})"

    def __add__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self._coeffs), len(other._coeffs))
        summed: list[Fraction] = []
        for i in range(n):
            a = self._coeffs[i] if i < len(self._coeffs) else Fraction(0)
            b = other._coeffs[i] if i < len(other._coeffs) else Fraction(0)
            summed.append(a + b)
        return Polynomial(summed)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self._coeffs), len(other._coeffs))
        diffed: list[Fraction] = []
        for i in range(n):
            a = self._coeffs[i] if i < len(self._coeffs) else Fraction(0)
            b = other._coeffs[i] if i < len(other._coeffs) else Fraction(0)
            diffed.append(a - b)
        return Polynomial(diffed)

    def __mul__(self, other) -> "Polynomial":
        if isinstance(other, Polynomial):
            if self.is_zero or other.is_zero:
                return Polynomial.zero()
            result = [Fraction(0)] * (len(self._coeffs) + len(other._coeffs) - 1)
            for i, a in enumerate(self._coeffs):
                for j, b in enumerate(other._coeffs):
                    result[i + j] += a * b
            return Polynomial(result)
        # Scalar multiplication: a number times every coefficient.
        scalar = _as_fraction_list([other])[0]
        return Polynomial([c * scalar for c in self._coeffs])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self) -> "Polynomial":
        return Polynomial([-c for c in self._coeffs])

    def divmod_by(self, divisor: "Polynomial"):
        """Exact polynomial division with remainder over :class:`~fractions.Fraction`.

        Returns ``(quotient, remainder)``. Raises :class:`ZeroDivisionError`
        if ``divisor`` is the zero polynomial.
        """
        if divisor.is_zero:
            raise ZeroDivisionError("polynomial division by zero")
        if self.degree < divisor.degree:
            return Polynomial.zero(), self

        quotient_coeffs = [Fraction(0)] * (self.degree - divisor.degree + 1)
        # We work on a mutable copy of the dividend's coefficients; the
        # algorithm subtracts scaled shifts of the divisor until what
        # remains has degree smaller than the divisor's.
        remainder_coeffs = list(self._coeffs)
        d_lead = divisor.leading_coefficient()
        d_deg = divisor.degree

        for i in range(self.degree - divisor.degree, -1, -1):
            if i + d_deg >= len(remainder_coeffs):
                continue
            coeff = remainder_coeffs[i + d_deg]
            if coeff == 0:
                quotient_coeffs[i] = Fraction(0)
                continue
            factor = coeff / d_lead
            quotient_coeffs[i] = factor
            for j in range(d_deg + 1):
                remainder_coeffs[i + j] -= factor * divisor._coeffs[j]

        quotient = Polynomial(quotient_coeffs)
        remainder = Polynomial(remainder_coeffs)
        return quotient, remainder

    def __floordiv__(self, other: "Polynomial") -> "Polynomial":
        q, _ = self.divmod_by(other)
        return q

    def __mod__(self, other: "Polynomial") -> "Polynomial":
        _, r = self.divmod_by(other)
        return r

    def monic(self) -> "Polynomial":
        """Return this polynomial scaled so its leading coefficient is 1.

        The zero polynomial returns itself, since it has no leading
        coefficient to normalise.
        """
        if self.is_zero:
            return self
        lc = self.leading_coefficient()
        return Polynomial([c / lc for c in self._coeffs])


def _gcd_step(a: Polynomial, b: Polynomial) -> Polynomial:
    """One iteration of the Euclidean loop: remainder of ``a`` divided by ``b``."""
    _, r = a.divmod_by(b)
    return r


def gcd(a: Polynomial, b: Polynomial) -> Polynomial:
    """Greatest common divisor of two rational polynomials, made monic.

    We run the Euclidean algorithm exactly: at each step the remainder is a
    polynomial over :class:`~fractions.Fraction`, so there is no rounding and
    no coefficient swell beyond what the numerator/denominator integers
    themselves grow to.

    The result is normalised to be monic (leading coefficient 1). This makes
    the GCD unique up to multiplication by a non-zero rational, which is the
    standard convention for polynomial GCDs over a field.

    Edge cases:
      * ``gcd(0, 0)`` returns the zero polynomial.
      * ``gcd(0, p)`` returns ``p`` made monic (or zero if ``p`` is zero).
    """
    if not isinstance(a, Polynomial) or not isinstance(b, Polynomial):
        raise TypeError("gcd() arguments must be Polynomial instances")

    x, y = a, b
    while not y.is_zero:
        x, y = y, _gcd_step(x, y)

    if x.is_zero:
        return Polynomial.zero()
    return x.monic()
