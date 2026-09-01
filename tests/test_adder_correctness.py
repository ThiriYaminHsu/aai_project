"""Unitary-simulation correctness test for ripple_carry_adder.

Exhaustively checks, for n = 1..3, that the generated circuit computes
B <- A + B (mod 2^n) with register A and the carry ancilla restored
(Cuccaro et al. 2004, Section 4.1 modulo-2^n family), and that setting
the ancilla to 1 yields A + B + 1 (their Section 4.2 incoming-carry
behaviour). Added 2026-08-31 after unitary simulation caught a
carry-wiring bug in the original _maj/_uma implementation.

Run:  python3 tests/test_adder_correctness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from circopt_adder.generators import ripple_carry_adder


def check(n: int) -> None:
    circ = ripple_carry_adder(n)
    qc = QuantumCircuit.from_qasm_str(circ.to_basic_gates().to_qasm())
    U = Operator(qc).data
    N = 2 ** n
    for a in range(N):
        for b in range(N):
            for anc_in, expect_b in ((0, (a + b) % N), (1, (a + b + 1) % N)):
                idx = anc_in | (a << 1) | (b << (n + 1))
                col = U[:, idx]
                out = int(np.argmax(np.abs(col)))
                assert abs(col[out]) > 0.999, f"n={n}: not a permutation at {(a, b, anc_in)}"
                assert (out >> 1) & (N - 1) == a, f"n={n}: A not restored at {(a, b, anc_in)}"
                assert (out >> (n + 1)) & (N - 1) == expect_b, f"n={n}: wrong sum at {(a, b, anc_in)}"
                if anc_in == 0:
                    assert out & 1 == 0, f"n={n}: ancilla not restored at {(a, b)}"


if __name__ == "__main__":
    for n in (1, 2, 3):
        check(n)
        print(f"n={n}: OK")
    print("ripple_carry_adder verified: B <- A + B (mod 2^n), A and ancilla restored")
