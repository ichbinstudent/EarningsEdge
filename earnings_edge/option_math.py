"""
Black-Scholes pricing, implied-volatility solver, and greeks.

Extracted from analyzer.py so both the analyzer and the Polygon
market-data provider can share the same math without a circular import.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """European option price (Black-Scholes-Merton)."""
    if T <= 0 or sigma <= 0:
        if T == 0:
            return max(0.0, (S - K) if option_type == "call" else (K - S))
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d1) - S * np.exp(-q * T) * norm.cdf(-d2)


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """Black-Scholes delta (nan when undefined)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(-np.exp(-q * T) * norm.cdf(-d1))


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """Black-Scholes gamma (same for calls and puts; nan when undefined)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """Black-Scholes theta per year (nan when undefined)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    decay = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return float(decay - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
    return float(decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """Black-Scholes vega per 1.00 of vol (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def black_scholes_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0,
) -> float:
    """Black-Scholes rho per 1.00 of rate (nan when undefined)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0,
    tol: float = 1e-6,
    vol_min: float = 1e-5,
    vol_max: float = 5.0,
) -> float:
    """Solve for the IV that makes BS price equal *market_price*."""
    def objective(sigma: float) -> float:
        try:
            return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price
        except (ValueError, ZeroDivisionError):
            return 1e10

    intrinsic = (
        max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        if option_type == "call"
        else max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
    )
    if market_price < intrinsic - tol:
        return np.nan

    try:
        min_price = black_scholes_price(S, K, T, r, vol_min, option_type, q)
        max_price = black_scholes_price(S, K, T, r, vol_max, option_type, q)
    except Exception:
        return np.nan

    if market_price < min_price - tol or market_price > max_price + tol:
        return np.nan

    try:
        return brentq(objective, vol_min, vol_max, xtol=tol, rtol=tol)
    except ValueError:
        return np.nan
