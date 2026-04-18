"""HyperLogLog sketch for cardinality / union estimation."""

import numpy as np


# ---------------------------------------------------------------------------
# Hash function (splitmix64 finaliser — good avalanche, no external deps)
# ---------------------------------------------------------------------------

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _hash64(x: int) -> int:
    """64-bit integer hash: splitmix64 finaliser."""
    x = x & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x = (x ^ (x >> 31)) & _MASK64
    return x


# ---------------------------------------------------------------------------
# HLL helpers
# ---------------------------------------------------------------------------

def _alpha(m: int) -> float:
    """Bias-correction constant for HLL."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


def _estimate_from_registers(registers: np.ndarray, alpha: float) -> float:
    """Compute HLL cardinality estimate from a register array."""
    m = len(registers)
    Z = float(np.sum(np.power(2.0, -registers.astype(np.float64))))
    E = alpha * m * m / Z

    # Small-range correction (LinearCounting)
    if E <= 2.5 * m:
        V = int(np.sum(registers == 0))
        if V > 0:
            E = m * np.log(m / V)

    # Large-range correction for the 64-bit hash domain used by _hash64.
    two64 = float(1 << 64)
    if E > two64 / 30.0:
        E = -two64 * np.log(1.0 - E / two64)

    return E


# ---------------------------------------------------------------------------
# HLL class
# ---------------------------------------------------------------------------

class HLL:
    """
    HyperLogLog sketch.

    Parameters
    ----------
    p : precision parameter; 2^p registers.  Typical values: 8–12.
        Relative std error ≈ 1.04 / sqrt(2^p).
    """

    def __init__(self, p: int = 8) -> None:
        if not (4 <= p <= 16):
            raise ValueError(f"p must be in [4, 16], got {p}")
        self.p = p
        self.m: int = 1 << p                  # number of registers
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._alpha = _alpha(self.m)
        self._remaining_bits: int = 64 - p   # bits used for rho computation

    # ------------------------------------------------------------------
    def add(self, x: int) -> None:
        """Add a single integer element."""
        h = _hash64(x)
        j = h >> self._remaining_bits                          # top-p bits → register index
        w = h & ((1 << self._remaining_bits) - 1)             # remaining bits
        if w == 0:
            rho = self._remaining_bits + 1
        else:
            rho = self._remaining_bits - w.bit_length() + 1   # position of leftmost 1
        if rho > self.registers[j]:
            self.registers[j] = rho

    def add_array(self, arr: np.ndarray) -> None:
        """Add all integer elements from a numpy array."""
        for x in arr:
            self.add(int(x))

    # ------------------------------------------------------------------
    def estimate(self) -> float:
        """Estimate the cardinality of the set."""
        return _estimate_from_registers(self.registers, self._alpha)

    def memory_bytes(self) -> int:
        """Raw register storage: 1 byte per register."""
        return self.m


# ---------------------------------------------------------------------------
# Union estimation
# ---------------------------------------------------------------------------

def union_estimate(hll1: HLL, hll2: HLL) -> float:
    """
    Estimate |A ∪ B| by merging two HLL sketches
    (take element-wise max of registers, then estimate).
    """
    if hll1.p != hll2.p:
        raise ValueError("Both HLLs must have the same precision p")
    merged = np.maximum(hll1.registers, hll2.registers)
    return _estimate_from_registers(merged, hll1._alpha)
