#!/usr/bin/env python3
"""
Genetics 1003: Chi-Square Calculator

This script calculates the chi-square statistic and p-value for:
1. Goodness-of-fit tests
2. 2×2 contingency table tests (independence/association)

It can handle both equal and specified expected frequencies, including special cases
where expected frequencies are zero.

For 2×2 contingency tables, it supports:
- Standard chi-square test
- Yates' continuity correction (for small expected frequencies)

Special handling for zero expected frequencies:
- If expected = 0 and observed = 0: contribution to chi-square is 0
- If expected = 0 and observed ≠ 0: chi-square statistic is infinity (reject hypothesis)

Author: AI Assistants
Date: 2025
"""

import math
from typing import Union, Optional, Sequence
import argparse


class ChiSquareCalculator:
    """A class to perform chi-square goodness-of-fit tests."""

    def __init__(
        self,
        observed: Sequence[Union[int, float]],
        expected: Optional[Sequence[Union[int, float]]] = None,
        expected_proportions: Optional[Sequence[float]] = None,
    ):
        """
        Initialize the calculator with observed and expected frequencies.

        Args:
            observed: List of observed frequencies
            expected: List of expected frequencies (optional)
            expected_proportions: List of expected proportions (optional)
        """
        self.observed = observed
        self.expected = expected
        self.expected_proportions = expected_proportions

        # Validate input data
        self._validate_input()

        # Calculate expected frequencies if not provided
        if self.expected is None:
            self._calculate_expected_frequencies()

    def _validate_input(self):
        """Validate the input data."""
        if not self.observed:
            raise ValueError("Observed frequencies cannot be empty")

        if any(freq < 0 for freq in self.observed):
            raise ValueError("All observed frequencies must be non-negative")

        if self.expected is not None:
            if len(self.observed) != len(self.expected):
                raise ValueError("Observed and expected frequencies must have the same length")
            if any(freq < 0 for freq in self.expected):
                raise ValueError("All expected frequencies must be non-negative")
            # Note: Zero expected frequencies are now allowed with special handling

        if self.expected_proportions is not None:
            if len(self.observed) != len(self.expected_proportions):
                raise ValueError("Observed frequencies and expected proportions must have the same length")
            if not math.isclose(sum(self.expected_proportions), 1.0, rel_tol=1e-9):
                raise ValueError("Expected proportions must sum to 1.0")

    def _calculate_expected_frequencies(self):
        """Calculate expected frequencies based on equal distribution or proportions."""
        total_observed = sum(self.observed)

        if self.expected_proportions is not None:
            # Use specified proportions
            self.expected = [total_observed * prop for prop in self.expected_proportions]
        else:
            # Assume equal distribution
            n_categories = len(self.observed)
            self.expected = [total_observed / n_categories] * n_categories

    def calculate_chi_square(self) -> float:
        """
        Calculate the chi-square statistic.

        Handles special cases where expected frequencies are zero:
        - If expected = 0 and observed = 0: contribution is 0
        - If expected = 0 and observed ≠ 0: contribution is positive infinity (reject hypothesis)

        Returns:
            The chi-square statistic value
        """
        chi_square = 0.0

        for obs, exp in zip(self.observed, self.expected or []):
            if exp == 0:
                if obs == 0:
                    # Both expected and observed are 0: contribution is 0
                    continue
                else:
                    # Expected is 0 but observed is not 0: return positive infinity
                    return float("inf")
            else:
                chi_square += ((obs - exp) ** 2) / exp

        return chi_square

    def calculate_degrees_of_freedom(self) -> int:
        """
        Calculate degrees of freedom.

        Returns:
            Degrees of freedom
        """
        return len(self.observed) - 1

    def _log_gamma(self, x: float) -> float:
        """Calculate log gamma function using Lanczos approximation."""
        if x < 0.5:
            # Reflection formula
            return math.log(math.pi) - math.log(math.sin(math.pi * x)) - self._log_gamma(1.0 - x)

        # Lanczos coefficients
        g = 7
        c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]

        x -= 1
        a = c[0]
        for i in range(1, g + 2):
            a += c[i] / (x + i)

        t = x + g + 0.5
        return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)

    def _regularized_gamma_p(self, s: float, x: float) -> float:
        """Calculate regularized incomplete gamma function P(s,x)."""
        if x == 0:
            return 0.0
        if x < 0:
            return 0.0

        # Use series expansion for small x
        if x < s + 1:
            # Series: P(s,x) = (x^s * e^(-x)) / Γ(s) * Σ(x^k / (s(s+1)...(s+k)))
            sum_term = 1.0
            term = 1.0
            for k in range(1, 100):
                term *= x / (s + k)
                sum_term += term
                if abs(term) < 1e-15:
                    break

            log_numerator = s * math.log(x) - x + math.log(sum_term)
            log_denominator = self._log_gamma(s)
            return math.exp(log_numerator - log_denominator)
        else:
            # Use continued fraction for large x
            # P(s,x) = 1 - Q(s,x) where Q(s,x) is the upper incomplete gamma
            return 1.0 - self._regularized_gamma_q(s, x)

    def _regularized_gamma_q(self, s: float, x: float) -> float:
        """Calculate regularized upper incomplete gamma function Q(s,x)."""
        if x <= 0:
            return 1.0  # Q(s,0) = 1

        # Continued fraction: Q(s,x) = (x^s * e^(-x)) / Γ(s) * 1/(x + 1-s/(x + 2-s/(x + ...)))

        # Lentz's algorithm for continued fraction
        tiny = 1e-30
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d

        for i in range(1, 100):
            a = -i * (s - i)
            b += 2.0
            d = 1.0 / (a * d + b)
            c = b + a / c
            if abs(c) < tiny:
                c = tiny
            delta = c * d
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break

        log_numerator = s * math.log(x) - x
        log_denominator = self._log_gamma(s)
        return math.exp(log_numerator - log_denominator) * h

    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """
        Calculate p-value for chi-square distribution using Wilson-Hilferty approximation.
        This is a reliable approximation for df >= 1.
        """
        if chi_square <= 0:
            return 1.0

        # Wilson-Hilferty transformation: (χ²/df)^(1/3) ~ N(1 - 2/(9df), 2/(9df))
        transformed = (chi_square / df) ** (1 / 3)
        mean = 1 - 2 / (9 * df)
        std_dev = math.sqrt(2 / (9 * df))

        # Standard normal CDF
        z_score = (transformed - mean) / std_dev
        return 1 - self._normal_cdf(z_score)

    def _normal_cdf(self, x: float) -> float:
        """Calculate standard normal CDF using error function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calculate_p_value(self, chi_square: float, df: int) -> float:
        """
        Calculate the p-value for the chi-square statistic.

        Args:
            chi_square: The chi-square statistic value
            df: Degrees of freedom

        Returns:
            The p-value
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")

        if chi_square < 0:
            raise ValueError("Chi-square statistic cannot be negative")

        # Handle infinity case (when expected=0 but observed≠0)
        if math.isinf(chi_square):
            return 0.0  # p-value is 0 for infinite chi-square statistic

        # Calculate p-value using chi-square approximation
        return self._chi_square_p_value(chi_square, df)

    def perform_test(self) -> dict:
        """
        Perform the complete chi-square goodness-of-fit test.

        Returns:
            Dictionary containing test results
        """
        chi_square = self.calculate_chi_square()
        df = self.calculate_degrees_of_freedom()
        p_value = self.calculate_p_value(chi_square, df)

        return {
            "chi_square_statistic": chi_square,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "observed_frequencies": self.observed,
            "expected_frequencies": self.expected,
            "significance_level": 0.05,  # Default significance level
            "result": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
        }

    def print_results(self, results: dict):
        """Print the test results in a formatted way."""
        print("\n" + "=" * 60)
        print("CHI-SQUARE GOODNESS-OF-FIT TEST RESULTS")
        print("=" * 60)

        print(f"Chi-square statistic: {results['chi_square_statistic']:.4f}")
        print(f"Degrees of freedom: {results['degrees_of_freedom']}")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Significance level (α): {results['significance_level']}")
        print(f"Conclusion: {results['result']}")

        print("\nObserved vs Expected Frequencies:")
        print("-" * 40)
        print("Category\tObserved\tExpected\tDifference")
        print("-" * 40)

        for i, (obs, exp) in enumerate(zip(results["observed_frequencies"], results["expected_frequencies"])):
            diff = obs - exp
            print(f"{i+1}\t\t{obs}\t\t{exp:.2f}\t\t{diff:.2f}")

        print("=" * 60)


class ContingencyTableCalculator:
    """A class to perform chi-square test for 2×2 contingency tables."""

    def __init__(self, table: Sequence[Sequence[Union[int, float]]]):
        """
        Initialize the calculator with a 2×2 contingency table.

        Args:
            table: A 2×2 list/array representing the contingency table
                   Format: [[a, b], [c, d]]
        """
        self.table = [list(row) for row in table]
        self._validate_input()
        self.a, self.b = self.table[0]
        self.c, self.d = self.table[1]

    def _validate_input(self):
        """Validate the input contingency table."""
        if len(self.table) != 2:
            raise ValueError("Contingency table must have exactly 2 rows")
        if len(self.table[0]) != 2 or len(self.table[1]) != 2:
            raise ValueError("Contingency table must have exactly 2 columns")

        for row in self.table:
            for value in row:
                if value < 0:
                    raise ValueError("All values in contingency table must be non-negative")

    def calculate_marginals(self) -> dict:
        """Calculate row and column marginal totals."""
        row1_total = self.a + self.b
        row2_total = self.c + self.d
        col1_total = self.a + self.c
        col2_total = self.b + self.d
        grand_total = self.a + self.b + self.c + self.d

        return {
            "row1_total": row1_total,
            "row2_total": row2_total,
            "col1_total": col1_total,
            "col2_total": col2_total,
            "grand_total": grand_total,
        }

    def calculate_expected(self) -> list:
        """Calculate expected frequencies for each cell."""
        marginals = self.calculate_marginals()
        n = marginals["grand_total"]

        if n == 0:
            raise ValueError("Grand total cannot be zero")

        expected = [
            [
                marginals["row1_total"] * marginals["col1_total"] / n,
                marginals["row1_total"] * marginals["col2_total"] / n,
            ],
            [
                marginals["row2_total"] * marginals["col1_total"] / n,
                marginals["row2_total"] * marginals["col2_total"] / n,
            ],
        ]

        return expected

    def calculate_chi_square(self) -> float:
        """
        Calculate the chi-square statistic for the 2×2 contingency table.

        Returns:
            The chi-square statistic value
        """
        expected = self.calculate_expected()
        chi_square = 0.0

        for i in range(2):
            for j in range(2):
                obs = self.table[i][j]
                exp = expected[i][j]

                if exp == 0:
                    if obs == 0:
                        continue
                    else:
                        return float("inf")
                else:
                    chi_square += ((obs - exp) ** 2) / exp

        return chi_square

    def calculate_chi_square_with_correction(self) -> float:
        """
        Calculate the chi-square statistic with Yates' continuity correction.
        Recommended when any expected frequency is less than 5.

        Returns:
            The chi-square statistic value with Yates' correction
        """
        expected = self.calculate_expected()
        chi_square = 0.0

        for i in range(2):
            for j in range(2):
                obs = self.table[i][j]
                exp = expected[i][j]

                if exp == 0:
                    if obs == 0:
                        continue
                    else:
                        return float("inf")
                else:
                    # Apply Yates' correction: subtract 0.5 from |O - E|
                    corrected_diff = abs(obs - exp) - 0.5
                    if corrected_diff < 0:
                        corrected_diff = 0
                    chi_square += (corrected_diff**2) / exp

        return chi_square

    def calculate_degrees_of_freedom(self) -> int:
        """
        Calculate degrees of freedom for 2×2 contingency table.

        Returns:
            Degrees of freedom (always 1 for 2×2 table)
        """
        return 1

    def _log_gamma(self, x: float) -> float:
        """Calculate log gamma function using Lanczos approximation."""
        if x < 0.5:
            return math.log(math.pi) - math.log(math.sin(math.pi * x)) - self._log_gamma(1.0 - x)

        g = 7
        c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]

        x -= 1
        a = c[0]
        for i in range(1, g + 2):
            a += c[i] / (x + i)

        t = x + g + 0.5
        return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)

    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Calculate p-value for chi-square distribution."""
        if chi_square <= 0:
            return 1.0

        transformed = (chi_square / df) ** (1 / 3)
        mean = 1 - 2 / (9 * df)
        std_dev = math.sqrt(2 / (9 * df))

        z_score = (transformed - mean) / std_dev
        return 1 - self._normal_cdf(z_score)

    def _normal_cdf(self, x: float) -> float:
        """Calculate standard normal CDF using error function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calculate_p_value(self, chi_square: float) -> float:
        """
        Calculate the p-value for the chi-square statistic.

        Args:
            chi_square: The chi-square statistic value

        Returns:
            The p-value
        """
        if chi_square < 0:
            raise ValueError("Chi-square statistic cannot be negative")

        if math.isinf(chi_square):
            return 0.0

        return self._chi_square_p_value(chi_square, 1)

    def should_use_correction(self) -> bool:
        """
        Check if Yates' continuity correction should be used.
        Recommended when any expected frequency is less than 5.

        Returns:
            True if correction should be used, False otherwise
        """
        expected = self.calculate_expected()
        for row in expected:
            for exp in row:
                if exp < 5:
                    return True
        return False

    def perform_test(self, use_correction: Optional[bool] = None) -> dict:
        """
        Perform the complete chi-square test for independence.

        Args:
            use_correction: Whether to use Yates' correction. If None, automatically
                          determines based on expected frequencies.

        Returns:
            Dictionary containing test results
        """
        expected = self.calculate_expected()

        # Determine if correction should be used
        if use_correction is None:
            use_correction = self.should_use_correction()

        if use_correction:
            chi_square = self.calculate_chi_square_with_correction()
        else:
            chi_square = self.calculate_chi_square()

        df = self.calculate_degrees_of_freedom()
        p_value = self.calculate_p_value(chi_square)
        marginals = self.calculate_marginals()

        return {
            "chi_square_statistic": chi_square,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "observed_table": self.table,
            "expected_table": expected,
            "marginals": marginals,
            "yates_correction_used": use_correction,
            "significance_level": 0.05,
            "result": (
                "Reject H0 (variables are associated)"
                if p_value < 0.05
                else "Fail to reject H0 (variables are independent)"
            ),
        }

    def print_results(self, results: dict):
        """Print the test results in a formatted way."""
        print("\n" + "=" * 60)
        print("CHI-SQUARE TEST FOR 2×2 CONTINGENCY TABLE")
        print("=" * 60)

        print(f"Chi-square statistic: {results['chi_square_statistic']:.4f}")
        if results["yates_correction_used"]:
            print("  (with Yates' continuity correction)")
        print(f"Degrees of freedom: {results['degrees_of_freedom']}")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Significance level (α): {results['significance_level']}")
        print(f"Conclusion: {results['result']}")

        print("\nObserved Frequencies:")
        print("-" * 40)
        obs = results["observed_table"]
        print("          Column 1    Column 2    Total")
        print(f"Row 1     {obs[0][0]:8.0f}    {obs[0][1]:8.0f}    {results['marginals']['row1_total']:8.0f}")
        print(f"Row 2     {obs[1][0]:8.0f}    {obs[1][1]:8.0f}    {results['marginals']['row2_total']:8.0f}")
        print(
            f"Total     {results['marginals']['col1_total']:8.0f}    {results['marginals']['col2_total']:8.0f}    {results['marginals']['grand_total']:8.0f}"
        )

        print("\nExpected Frequencies:")
        print("-" * 40)
        exp = results["expected_table"]
        print("          Column 1    Column 2")
        print(f"Row 1     {exp[0][0]:8.2f}    {exp[0][1]:8.2f}")
        print(f"Row 2     {exp[1][0]:8.2f}    {exp[1][1]:8.2f}")

        print("=" * 60)


class HardyWeinbergCalculator:
    """A class to test Hardy-Weinberg equilibrium using chi-square test."""

    def __init__(self, genotype_counts: dict):
        """
        Initialize the calculator with observed genotype counts.

        Args:
            genotype_counts: Dictionary with genotype counts
                           Format: {'AA': count, 'Aa': count, 'aa': count}
                           or {'dominant_homozygous': count, 'heterozygous': count, 'recessive_homozygous': count}
        """
        self.genotype_counts = genotype_counts
        self._validate_input()
        self._normalize_genotype_keys()

    def _validate_input(self):
        """Validate the input genotype counts."""
        if len(self.genotype_counts) != 3:
            raise ValueError("Must provide exactly 3 genotype counts")

        for count in self.genotype_counts.values():
            if count < 0:
                raise ValueError("All genotype counts must be non-negative")

        total = sum(self.genotype_counts.values())
        if total == 0:
            raise ValueError("Total count cannot be zero")

    def _normalize_genotype_keys(self):
        """Normalize genotype keys to standard format (AA, Aa, aa)."""
        keys = list(self.genotype_counts.keys())

        # Check if already in standard format
        standard_keys = {"AA", "Aa", "aa"}
        if set(keys) == standard_keys:
            return

        # If using descriptive names, map to standard format
        if len(keys) == 3:
            # Assume order: dominant_homozygous, heterozygous, recessive_homozygous
            values = list(self.genotype_counts.values())
            self.genotype_counts = {"AA": values[0], "Aa": values[1], "aa": values[2]}

    def calculate_allele_frequencies(self) -> dict:
        """
        Calculate allele frequencies from observed genotype counts.

        Returns:
            Dictionary with allele frequencies {'p': freq_A, 'q': freq_a}
        """
        n_AA = self.genotype_counts["AA"]
        n_Aa = self.genotype_counts["Aa"]
        n_aa = self.genotype_counts["aa"]

        total = n_AA + n_Aa + n_aa

        # Calculate allele frequencies using allele counting method
        # Frequency of A allele (p) = (2*AA + Aa) / (2*total)
        # Frequency of a allele (q) = (2*aa + Aa) / (2*total)
        freq_A = (2 * n_AA + n_Aa) / (2 * total)
        freq_a = (2 * n_aa + n_Aa) / (2 * total)

        return {"p": freq_A, "q": freq_a}

    def calculate_expected_genotypes(self) -> dict:
        """
        Calculate expected genotype frequencies under Hardy-Weinberg equilibrium.

        Returns:
            Dictionary with expected genotype counts {'AA': count, 'Aa': count, 'aa': count}
        """
        allele_freqs = self.calculate_allele_frequencies()
        p = allele_freqs["p"]
        q = allele_freqs["q"]

        total = sum(self.genotype_counts.values())

        # Under Hardy-Weinberg equilibrium:
        # Frequency of AA = p²
        # Frequency of Aa = 2pq
        # Frequency of aa = q²
        expected = {"AA": p * p * total, "Aa": 2 * p * q * total, "aa": q * q * total}

        return expected

    def calculate_chi_square(self) -> float:
        """
        Calculate the chi-square statistic for Hardy-Weinberg equilibrium test.

        Returns:
            The chi-square statistic value
        """
        expected = self.calculate_expected_genotypes()
        chi_square = 0.0

        for genotype in ["AA", "Aa", "aa"]:
            obs = self.genotype_counts[genotype]
            exp = expected[genotype]

            if exp == 0:
                if obs == 0:
                    continue
                else:
                    return float("inf")
            else:
                chi_square += ((obs - exp) ** 2) / exp

        return chi_square

    def calculate_degrees_of_freedom(self) -> int:
        """
        Calculate degrees of freedom for Hardy-Weinberg test.

        For Hardy-Weinberg equilibrium test:
        df = number of genotypes - number of alleles = 3 - 2 = 1

        Returns:
            Degrees of freedom (always 1 for two-allele system)
        """
        return 1

    def _log_gamma(self, x: float) -> float:
        """Calculate log gamma function using Lanczos approximation."""
        if x < 0.5:
            return math.log(math.pi) - math.log(math.sin(math.pi * x)) - self._log_gamma(1.0 - x)

        g = 7
        c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]

        x -= 1
        a = c[0]
        for i in range(1, g + 2):
            a += c[i] / (x + i)

        t = x + g + 0.5
        return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)

    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Calculate p-value for chi-square distribution."""
        if chi_square <= 0:
            return 1.0

        transformed = (chi_square / df) ** (1 / 3)
        mean = 1 - 2 / (9 * df)
        std_dev = math.sqrt(2 / (9 * df))

        z_score = (transformed - mean) / std_dev
        return 1 - self._normal_cdf(z_score)

    def _normal_cdf(self, x: float) -> float:
        """Calculate standard normal CDF using error function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calculate_p_value(self, chi_square: float) -> float:
        """
        Calculate the p-value for the chi-square statistic.

        Args:
            chi_square: The chi-square statistic value

        Returns:
            The p-value
        """
        if chi_square < 0:
            raise ValueError("Chi-square statistic cannot be negative")

        if math.isinf(chi_square):
            return 0.0

        return self._chi_square_p_value(chi_square, 1)

    def perform_test(self) -> dict:
        """
        Perform the complete Hardy-Weinberg equilibrium test.

        Returns:
            Dictionary containing test results
        """
        allele_freqs = self.calculate_allele_frequencies()
        expected = self.calculate_expected_genotypes()
        chi_square = self.calculate_chi_square()
        df = self.calculate_degrees_of_freedom()
        p_value = self.calculate_p_value(chi_square)

        return {
            "chi_square_statistic": chi_square,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "observed_genotypes": self.genotype_counts,
            "expected_genotypes": expected,
            "allele_frequencies": allele_freqs,
            "significance_level": 0.05,
            "result": (
                "Reject H0 (not in Hardy-Weinberg equilibrium)"
                if p_value < 0.05
                else "Fail to reject H0 (in Hardy-Weinberg equilibrium)"
            ),
        }

    def print_results(self, results: dict):
        """Print the test results in a formatted way."""
        print("\n" + "=" * 60)
        print("HARDY-WEINBERG EQUILIBRIUM TEST")
        print("=" * 60)

        print(f"Chi-square statistic: {results['chi_square_statistic']:.4f}")
        print(f"Degrees of freedom: {results['degrees_of_freedom']}")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Significance level (α): {results['significance_level']}")
        print(f"Conclusion: {results['result']}")

        print("\nAllele Frequencies:")
        print("-" * 40)
        p = results["allele_frequencies"]["p"]
        q = results["allele_frequencies"]["q"]
        print(f"Frequency of A allele (p): {p:.4f}")
        print(f"Frequency of a allele (q): {q:.4f}")
        print(f"p + q = {p + q:.4f}")

        print("\nGenotype Frequencies:")
        print("-" * 40)
        print("Genotype    Observed    Expected    Difference")
        print("-" * 40)

        for genotype in ["AA", "Aa", "aa"]:
            obs = results["observed_genotypes"][genotype]
            exp = results["expected_genotypes"][genotype]
            diff = obs - exp
            print(f"{genotype:8}    {obs:8.0f}    {exp:8.2f}    {diff:8.2f}")

        total_obs = sum(results["observed_genotypes"].values())
        total_exp = sum(results["expected_genotypes"].values())
        print("-" * 40)
        print(f"Total       {total_obs:8.0f}    {total_exp:8.2f}")

        print("\nExpected Hardy-Weinberg Proportions:")
        print(f"  AA (p^2):  {p*p:.4f}")
        print(f"  Aa (2pq):  {2*p*q:.4f}")
        print(f"  aa (q^2):  {q*q:.4f}")

        print("=" * 60)


def main():
    """Main function to run the chi-square calculator."""
    parser = argparse.ArgumentParser(
        description="Chi-Square Calculator (Goodness-of-Fit, 2×2 Contingency Table, and Hardy-Weinberg Equilibrium)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Goodness-of-Fit: Equal expected frequencies
  python chi_square_calculator.py -o 10 15 20 25

  # Goodness-of-Fit: Specified expected frequencies
  python chi_square_calculator.py -o 10 15 20 25 -e 12 14 18 26

  # Goodness-of-Fit: Specified expected proportions
  python chi_square_calculator.py -o 45 55 -p 0.5 0.5

  # 2×2 Contingency Table
  python chi_square_calculator.py -t 20 10 5 15
  (Format: a b c d for table [[a, b], [c, d]])

  # 2×2 Contingency Table with Yates' correction
  python chi_square_calculator.py -t 20 10 5 15 --yates
  
  # Hardy-Weinberg Equilibrium Test
  python chi_square_calculator.py --hw 50 40 10
  (Format: AA Aa aa counts)
        """,
    )

    parser.add_argument("-o", "--observed", nargs="+", type=float, help="Observed frequencies for goodness-of-fit test")

    parser.add_argument("-e", "--expected", nargs="+", type=float, help="Expected frequencies")

    parser.add_argument("-p", "--proportions", nargs="+", type=float, help="Expected proportions (must sum to 1.0)")

    parser.add_argument("-t", "--table", nargs=4, type=float, help="2×2 contingency table values (a b c d)")

    parser.add_argument("--yates", action="store_true", help="Use Yates' continuity correction for 2×2 table")

    parser.add_argument(
        "--hw",
        "--hardy-weinberg",
        nargs=3,
        type=float,
        dest="hardy_weinberg",
        help="Hardy-Weinberg equilibrium test (provide counts for AA, Aa, aa)",
    )

    parser.add_argument("-a", "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")

    args = parser.parse_args()

    try:
        # Check if using Hardy-Weinberg equilibrium test
        if args.hardy_weinberg:
            genotype_counts = {"AA": args.hardy_weinberg[0], "Aa": args.hardy_weinberg[1], "aa": args.hardy_weinberg[2]}
            calculator = HardyWeinbergCalculator(genotype_counts=genotype_counts)

            # Perform test
            results = calculator.perform_test()
            results["significance_level"] = args.alpha
            results["result"] = (
                "Reject H0 (not in Hardy-Weinberg equilibrium)"
                if results["p_value"] < args.alpha
                else "Fail to reject H0 (in Hardy-Weinberg equilibrium)"
            )

            # Print results
            calculator.print_results(results)

        # Check if using 2×2 contingency table mode
        elif args.table:
            # Create 2×2 table
            table = [[args.table[0], args.table[1]], [args.table[2], args.table[3]]]
            calculator = ContingencyTableCalculator(table=table)

            # Perform test
            results = calculator.perform_test(use_correction=args.yates if args.yates else None)
            results["significance_level"] = args.alpha
            results["result"] = (
                "Reject H0 (variables are associated)"
                if results["p_value"] < args.alpha
                else "Fail to reject H0 (variables are independent)"
            )

            # Print results
            calculator.print_results(results)

        elif args.observed:
            # Goodness-of-fit test mode
            calculator = ChiSquareCalculator(
                observed=args.observed, expected=args.expected, expected_proportions=args.proportions
            )

            # Perform test
            results = calculator.perform_test()
            results["significance_level"] = args.alpha
            results["result"] = "Reject H0" if results["p_value"] < args.alpha else "Fail to reject H0"

            # Print results
            calculator.print_results(results)

        else:
            parser.error("Either -o/--observed, -t/--table, or --hw/--hardy-weinberg must be provided")

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


def run_examples():
    """Run some example calculations to demonstrate the calculator."""
    print("Running example calculations...\n")

    # Example 1: Testing if a die is fair
    print("Example 1: Testing if a die is fair")
    print("H0: The die is fair (equal probability for each face)")
    print("H1: The die is not fair")

    observed_die = [8, 12, 10, 15, 7, 8]  # Results from 60 rolls
    calculator1 = ChiSquareCalculator(observed=observed_die)
    results1 = calculator1.perform_test()
    calculator1.print_results(results1)

    # Example 2: Testing genetic ratios
    print("\nExample 2: Testing genetic ratios (3:1 ratio)")
    print("H0: The ratio is 3:1")
    print("H1: The ratio is not 3:1")

    observed_genetic = [180, 60]  # Observed counts
    proportions_genetic = [0.75, 0.25]  # Expected 3:1 ratio
    calculator2 = ChiSquareCalculator(observed=observed_genetic, expected_proportions=proportions_genetic)
    results2 = calculator2.perform_test()
    calculator2.print_results(results2)

    # Example 3: Testing specific expected frequencies
    print("\nExample 3: Testing specific expected frequencies")
    observed_custom = [25, 30, 45]
    expected_custom = [20, 35, 45]
    calculator3 = ChiSquareCalculator(observed=observed_custom, expected=expected_custom)
    results3 = calculator3.perform_test()
    calculator3.print_results(results3)

    # Example 4: Testing zero expected frequencies (both zero)
    print("\nExample 4: Testing zero expected frequencies (both zero)")
    print("H0: Expected frequency is 0, observed is also 0")
    print("H1: Not applicable - special case")

    observed_zero = [10, 0, 15, 20]
    expected_zero = [12, 0, 14, 19]
    calculator4 = ChiSquareCalculator(observed=observed_zero, expected=expected_zero)
    results4 = calculator4.perform_test()
    calculator4.print_results(results4)

    # Example 5: Testing zero expected frequencies (observed not zero)
    print("\nExample 5: Testing zero expected frequencies (observed not zero)")
    print("H0: Expected frequency is 0, but observed is not 0")
    print("H1: Should reject H0 (infinite chi-square)")

    observed_nonzero = [10, 5, 15, 20]
    expected_nonzero = [12, 0, 14, 19]
    calculator5 = ChiSquareCalculator(observed=observed_nonzero, expected=expected_nonzero)
    results5 = calculator5.perform_test()
    calculator5.print_results(results5)

    # Example 6: 2×2 Contingency Table - Treatment effectiveness
    print("\nExample 6: 2×2 Contingency Table - Testing treatment effectiveness")
    print("H0: Treatment and outcome are independent")
    print("H1: Treatment and outcome are associated")
    print("Table: [[Treatment Success, Treatment Failure], [Control Success, Control Failure]]")

    table1 = [[60, 40], [30, 70]]  # Treatment group vs Control group
    contingency_calc1 = ContingencyTableCalculator(table=table1)
    contingency_results1 = contingency_calc1.perform_test()
    contingency_calc1.print_results(contingency_results1)

    # Example 7: 2×2 Contingency Table with small expected frequencies
    print("\nExample 7: 2×2 Contingency Table - Small sample (Yates' correction)")
    print("H0: Gender and preference are independent")
    print("H1: Gender and preference are associated")

    table2 = [[8, 12], [4, 6]]  # Small sample size
    contingency_calc2 = ContingencyTableCalculator(table=table2)
    contingency_results2 = contingency_calc2.perform_test()
    contingency_calc2.print_results(contingency_results2)

    # Example 8: Hardy-Weinberg Equilibrium Test - In equilibrium
    print("\nExample 8: Hardy-Weinberg Equilibrium Test - Population in equilibrium")
    print("H0: Population is in Hardy-Weinberg equilibrium")
    print("H1: Population is not in Hardy-Weinberg equilibrium")

    genotypes1 = {"AA": 36, "Aa": 48, "aa": 16}  # Follows HW equilibrium (p=0.6, q=0.4)
    hw_calc1 = HardyWeinbergCalculator(genotype_counts=genotypes1)
    hw_results1 = hw_calc1.perform_test()
    hw_calc1.print_results(hw_results1)

    # Example 9: Hardy-Weinberg Equilibrium Test - Not in equilibrium
    print("\nExample 9: Hardy-Weinberg Equilibrium Test - Population not in equilibrium")
    print("H0: Population is in Hardy-Weinberg equilibrium")
    print("H1: Population is not in Hardy-Weinberg equilibrium")

    genotypes2 = {"AA": 50, "Aa": 30, "aa": 20}  # Deviates from HW equilibrium
    hw_calc2 = HardyWeinbergCalculator(genotype_counts=genotypes2)
    hw_results2 = hw_calc2.perform_test()
    hw_calc2.print_results(hw_results2)


if __name__ == "__main__":
    # If no arguments are provided, run examples
    import sys

    if len(sys.argv) == 1:
        run_examples()
    else:
        exit(main())
