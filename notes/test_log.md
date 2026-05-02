# SPN Test Log

## Purpose

This file records the current test coverage and validation milestones for the SPN Paper 2 modelling codebase.

The test suite is intended to lock down implementation conventions before further modelling, refactoring, or paper drafting. In particular, the tests protect against:

- sign errors;
- factor-of-two errors;
- incorrect low-sector gap conventions;
- inconsistent handling of `k_vector` and `||k_vector||`;
- loss of unitarity or norm preservation;
- confusion between microscopic transport and reduced packet-centre motion;
- accidental breakage of diagnostics used in notebooks and Paper 2 exploration.

---

## Current tested modules

The following source modules now have first-pass test coverage:

| Source file | Test file | Coverage focus | Status |
|---|---|---|---|
| `src/spn/kernels.py` | `tests/test_kernels.py` | `L^2 -> l(l+1)`, generator families, low-sector mean/gap, unitary kernel values | Passing |
| `src/spn/transport.py` | `tests/test_transport.py` | direction vectors, displacement, backward-shifted transport position, momentum-space phase, microscopic speed, coarse path diagnostics | Passing |
| `src/spn/evolution.py` | `tests/test_evolution.py` | reduced state validation, Hermiticity/unitarity, reduced Hamiltonian, group velocity, one-tick and n-tick evolution, norm preservation | Passing |
| `src/spn/reduced_models.py` | `tests/test_reduced_models.py` | model config, defensive copies, Hamiltonian/unitary wrapper, group velocity, state construction, evolution, diagnostics | Passing |
| `src/spn/observables.py` | `tests/test_observables.py` | spectral gaps, eigensystems, expectations, Bloch diagnostics, sector weights, leakage, coupling-to-gap ratios | Passing |
| `src/spn/topology.py` | `tests/test_topology.py` | phase wrapping, phase differences, winding diagnostics, complex path winding, spinor relative phase diagnostics | Passing |
| `src/spn/utils.py` | `tests/test_utils.py` | finite array validation, scalar checks, vector/matrix helpers, Hermitian/unitary checks, probability weights, array comparisons | Passing |

---

## Key conventions now protected by tests

### Kernel conventions

- `l2_eigenvalue(l)` returns:

  ```text
  l(l+1)
  ```

- `generator_laplace_beltrami(l, alpha)` scales the `L^2` eigenvalue.
- `generator_linear(l, kappa)` is linear in `L^2`, not linear in `l`.
- `generator_poly2(l, a, b)` implements a quadratic polynomial in `L^2`.
- `low_sector_gap(...)` returns the Paper-2 half-gap:

  ```text
  Delta = (f(1) - f(0)) / 2
  ```

  not the full gap.

### Transport conventions

- `unit_direction(theta, phi)` uses the convention:

  ```text
  x = sin(theta) cos(phi)
  y = sin(theta) sin(phi)
  z = cos(theta)
  ```

- `transport_displacement(theta, phi, L)` returns:

  ```text
  L * n_hat(Omega)
  ```

- `transported_position(x, theta, phi, L)` uses the configuration-space backward shift:

  ```text
  x -> x - L * n_hat(Omega)
  ```

- `transport_phase(k, theta, phi, L)` returns:

  ```text
  exp(-i k . (L n_hat(Omega)))
  ```

### Reduced Paper 2 evolution conventions

The reduced axisymmetric model uses:

```text
H_red = fbar I + Delta sigma_z + v k sigma_x
```

where:

```text
fbar  = (f(1) + f(0)) / 2
Delta = (f(1) - f(0)) / 2
v     = L / (tau sqrt(3))
k     = ||k_vector||
```

The reduced packet centre evolves by group velocity:

```text
x_{t+1} = x_t + tau * v_group
```

This is distinct from the full SPN microscopic transport rule.

### Diagnostic conventions

- Spinor norm should be preserved under reduced unitary evolution up to numerical precision.
- Bloch radius for pure two-level states should remain approximately 1.
- Sector diagnostics assume contiguous basis blocks.
- `coupling_to_gap_ratio` returns:

  ```text
  coupling_norm / |E_j - E_i|
  ```

- `gap_to_coupling_ratio` is retained only as a backwards-compatible alias and returns the same value.

---

## Notebook smoke-test results

The first reduced-model notebook smoke test confirmed:

- package imports worked after adding the repo `src/` path;
- model construction worked;
- diagnostics worked;
- spinor norm drift stayed at numerical precision;
- Bloch radius remained essentially 1;
- measured packet-centre speed matched analytic group velocity;
- direct formula comparison matched the model exactly for sampled values.

### k-sweep results

The notebook k-sweep compared:

1. measured packet speed from trajectory displacement;
2. model diagnostic group velocity;
3. direct analytic formula from the dispersion relation.

Recorded results:

```text
max measured-vs-analytic speed error: 1.2212453270876722e-15
max formula difference: 0.0
```

The reduced group speed stayed below the reduced coupling-speed limit:

```text
reduced coupling speed limit: 0.5773502691896258
max group speed in sweep:     0.5547001962252291
below limit:                  True
```

This supports the implementation of:

```text
v_g(k) = v^2 k / sqrt(Delta^2 + v^2 k^2)
```

with asymptotic bound:

```text
v_g(k) -> v = L / (tau sqrt(3))
```

---

## Test workflow

Standard command from the repo root:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Individual test files can be run as:

```bash
PYTHONPATH=src python -m pytest tests/test_kernels.py -q
PYTHONPATH=src python -m pytest tests/test_transport.py -q
PYTHONPATH=src python -m pytest tests/test_evolution.py -q
PYTHONPATH=src python -m pytest tests/test_reduced_models.py -q
PYTHONPATH=src python -m pytest tests/test_observables.py -q
PYTHONPATH=src python -m pytest tests/test_topology.py -q
PYTHONPATH=src python -m pytest tests/test_utils.py -q
```

---

## Git workflow used

After each test file passed, changes were committed and pushed to GitHub.

General pattern:

```bash
git status
git add tests/<test_file>.py
git commit -m "Add <module> tests"
git push
```

For source changes plus tests:

```bash
git add src/spn/<module>.py tests/<test_file>.py
git commit -m "Add <module> tests"
git push
```

---

## Recommended next pressure tests

The current tests are implementation and internal-consistency tests. The next stage should put pressure on the reduced Paper-2 model itself:

1. **Isotropy stress test**
   - Hold `||k_vector||` fixed while varying direction.
   - Check that energy and group-speed magnitude are invariant.
   - `tests/test_isotropy.py` — Paper-2 isotropy pressure tests: fixed-||k|| Hamiltonian invariance, energy invariance, group-speed isotropy, branch direction conventions — Passing

2. **Low-k massive/Dirac-like dispersion fit**
   - Fit `E_+(k)` against `k^2` for small `k`.
   - Check agreement with the expected low-k expansion.
   - `tests/test_low_k_dispersion.py` — Paper-2 low-k massive/Dirac-like dispersion tests: exact reduced dispersion, small-k positive/negative branch quadratic expansion, fitted positive-branch coefficient, linear low-k group velocity, and k=0 half-gap convention — Passing

3. **Generator-family robustness**
   - Compare Laplace-Beltrami, linear-in-`L^2`, and polynomial generator families.
   - Check unitary stability, energy gap behaviour, and bounded group speed.
   - `tests/test_generator_family_robustness.py` — Paper-2 generator-family robustness tests: Laplace–Beltrami, linear-in-\(L^2\), and polynomial generators preserve finite nonzero low-sector half-gap, Hermiticity, unitarity, exact dispersion, isotropy, bounded reduced group speed, branch direction conventions, and low-\(k\) quadratic behaviour — Passing

4. **Long-time stability**
   - Evolve for `10_000` or more ticks.
   - Track norm drift, Bloch radius drift, and packet-centre linearity.
   - `tests/test_long_time_stability.py` — Paper-2 long-time numerical stability tests over 10,000 model ticks: preserves spinor norm, packet-centre motion matches analytic reduced group velocity, per-tick packet displacement remains constant, reduced packet speed stays below `L/(tau sqrt(3))`, and no-history evolution returns the correct final state. — Passing

5. **Enlarged-sector leakage test**
   - Move beyond the two-level reduced model.
   - Construct a 4-level or larger truncation.
   - Measure low-sector retention and leakage under controlled inter-sector coupling.
   - `tests/test_enlarged_sector_leakage.py` — Paper-2 enlarged-sector leakage / reduction-stability toy tests: embeds the tested 2D reduced Hamiltonian into a controlled 4D low-plus-complement model, verifies the low block matches the current reduced Hamiltonian, verifies exact block closure at zero coupling up to numerical precision, checks agreement with the original two-level reduced unitary when decoupled, checks total norm preservation, and confirms peak leakage increases with stronger low-complement coupling or smaller complement-sector separation across representative low-sector initial states. This is a controlled toy enlarged-sector pressure test, not a derivation of full \(L^2(S^2)\) sector stability — Passing

The enlarged-sector leakage test is the most important conceptual pressure test because it asks whether the two-level reduction is approximately stable rather than merely convenient.

---

## Current status

The codebase has a complete first-pass tested base across the main current modules:

```text
kernels.py
transport.py
evolution.py
reduced_models.py
observables.py
topology.py
utils.py
```

The project is ready for notebook-based pressure tests and subsequent Paper 2 modelling analysis.
