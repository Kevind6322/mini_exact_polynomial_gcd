import unittest
from fractions import Fraction

from mini_exact_polynomial_gcd import Polynomial, gcd


class TestPolynomialConstruction(unittest.TestCase):

    def test_empty_is_zero_polynomial(self):
        p = Polynomial([])
        self.assertTrue(p.is_zero)
        self.assertEqual(p.degree, -1)

    def test_trailing_zeros_stripped(self):
        p = Polynomial([Fraction(1), Fraction(0), Fraction(0)])
        self.assertEqual(p.degree, 0)
        self.assertEqual(p.coefficients, [Fraction(1)])

    def test_int_coefficients_accepted(self):
        p = Polynomial([1, -2, 3])
        self.assertEqual(p.coefficients, [Fraction(1), Fraction(-2), Fraction(3)])

    def test_float_coefficients_converted_via_string(self):
        # 0.5 should become exactly 1/2, not the binary-float approximation.
        p = Polynomial([0.5])
        self.assertEqual(p.coefficients, [Fraction(1, 2)])


class TestPolynomialArithmetic(unittest.TestCase):

    def test_addition(self):
        a = Polynomial([1, 2, 3])
        b = Polynomial([4, 5])
        result = a + b
        self.assertEqual(result.coefficients, [Fraction(5), Fraction(7), Fraction(3)])

    def test_subtraction(self):
        a = Polynomial([5, 7, 3])
        b = Polynomial([4, 5])
        result = a - b
        self.assertEqual(result.coefficients, [Fraction(1), Fraction(2), Fraction(3)])

    def test_multiplication_by_polynomial(self):
        a = Polynomial([1, 1])  # x + 1
        b = Polynomial([1, -1])  # x - 1
        result = a * b
        self.assertEqual(result.coefficients, [Fraction(1), Fraction(0), Fraction(-1)])

    def test_multiplication_by_scalar(self):
        a = Polynomial([1, 2, 3])
        result = a * Fraction(2)
        self.assertEqual(result.coefficients, [Fraction(2), Fraction(4), Fraction(6)])

    def test_scalar_left_multiply(self):
        a = Polynomial([1, 2, 3])
        result = 3 * a
        self.assertEqual(result.coefficients, [Fraction(3), Fraction(6), Fraction(9)])


class TestDivision(unittest.TestCase):

    def test_clean_division(self):
        dividend = Polynomial([1, 0, 0, 1])  # x^3 + 1
        divisor = Polynomial([1, 1])  # x + 1
        q, r = dividend.divmod_by(divisor)
        self.assertEqual(r, Polynomial.zero())
        self.assertEqual(q.coefficients, [Fraction(1), Fraction(-1), Fraction(1)])

    def test_division_with_remainder(self):
        dividend = Polynomial([1, 0, 0, 1])  # x^3 + 1
        divisor = Polynomial([1, 0, 1])  # x^2 + 1
        q, r = dividend.divmod_by(divisor)
        self.assertEqual(q.coefficients, [Fraction(0), Fraction(1)])
        self.assertEqual(r.coefficients, [Fraction(1), Fraction(-1)])

    def test_division_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Polynomial([1, 2]).divmod_by(Polynomial.zero())


class TestGCD(unittest.TestCase):

    def test_common_linear_factor(self):
        # (x-1)(x+2) and (x-1)(x-3)
        a = Polynomial([-2, 1, 1])
        b = Polynomial([3, -4, 1])
        result = gcd(a, b)
        self.assertEqual(result, Polynomial([-1, 1]))

    def test_coprime_polynomials(self):
        a = Polynomial([1, 1])  # x + 1
        b = Polynomial([1, -1])  # x - 1
        result = gcd(a, b)
        self.assertEqual(result, Polynomial([1]))

    def test_result_is_monic(self):
        a = Polynomial([4, 2])  # 2x + 4 = 2(x + 2)
        b = Polynomial([6, 3])  # 3x + 6 = 3(x + 2)
        result = gcd(a, b)
        self.assertEqual(result, Polynomial([2, 1]))

    def test_gcd_with_zero(self):
        p = Polynomial([4, 2])  # 2x + 4
        result = gcd(p, Polynomial.zero())
        self.assertEqual(result, Polynomial([2, 1]))

    def test_gcd_of_zero_and_zero(self):
        result = gcd(Polynomial.zero(), Polynomial.zero())
        self.assertTrue(result.is_zero)

    def test_gcd_with_constant(self):
        p = Polynomial([1, 2, 1])  # (x+1)^2
        c = Polynomial([5])
        result = gcd(p, c)
        self.assertEqual(result, Polynomial([1]))


if __name__ == "__main__":
    unittest.main()
