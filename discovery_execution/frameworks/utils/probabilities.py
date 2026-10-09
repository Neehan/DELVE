"""Reusable Beta-binomial likelihood and closed-form Beta/Dirichlet expected products."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, digamma, poch


def beta_binomial_nll(
    log_shapes: NDArray[np.float64], successes: NDArray[np.float64], attempts: NDArray[np.float64],
) -> list[float | NDArray[np.float64]]:
    """Return -sum_i[log B(a+u_i,b+m_i-u_i)-log B(a,b)] and its gradient with respect to log(a),log(b).

    u_i=successes[i], m_i=attempts[i], and B is the beta function. This integrates q_i^u_i*(1-q_i)^(m_i-u_i) under q_i~Beta(a,b); data-only binomial coefficients are omitted.
    """
    a, b = np.exp(log_shapes)
    failures = attempts - successes
    # Binomial coefficients do not depend on a or b, so they can be omitted when fitting.
    value = -np.sum(betaln(a + successes, b + failures) - betaln(a, b))
    total_change = digamma(a + b + attempts) - digamma(a + b)
    gradient = -np.array([
        a * np.sum(digamma(a + successes) - digamma(a) - total_change),
        b * np.sum(digamma(b + failures) - digamma(b) - total_change),
    ])
    return [float(value), gradient]


def expected_beta_product(
    a: NDArray[np.float64], b: NDArray[np.float64], successes: int, failures: int,
) -> NDArray[np.float64]:
    """Return E[q^s*(1-q)^f]=B(a+s,b+f)/B(a,b) for q~Beta(a,b), s=successes, f=failures.

    Evaluate the equivalent ratio poch(a,s)*poch(b,f)/poch(a+b,s+f), where poch(x,r)=x*(x+1)*...*(x+r-1). Arrays a and b broadcast over problems and failure-count cases.
    """
    return poch(a, successes) * poch(b, failures) / poch(a + b, successes + failures)


def expected_dirichlet_product(
    concentrations: NDArray[np.float64], powers: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return E[product_j pi_j^r_j]=product_j poch(d_j,r_j)/poch(sum_j d_j,sum_j r_j) for pi~Dirichlet(d).

    concentrations contains one row of d_j per problem; powers contains the nonnegative integer exponents r_j. poch is the rising factorial. Zero concentrations denote excluded categories; a positive exponent there makes the product zero.
    """
    numerator = np.prod(poch(concentrations, powers), axis=1)
    denominator = poch(concentrations.sum(axis=1), int(powers.sum()))
    return numerator / denominator
