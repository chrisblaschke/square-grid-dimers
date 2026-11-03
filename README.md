# Uniform square-grid dimers versus monotone and nonnegative circuits

Chris Blaschke (Ax0s), November 3, 2026. Preprint, not peer reviewed. [Read the paper (PDF).](paper/square-grid-dimers.pdf)

In 1980 Valiant proved that monotone arithmetic circuits for the perfect-matching polynomial of a triangular grid need exponential size. At the end of that paper he noted that one of his lemmas fails for rectangular grids, so their monotone complexity was left unresolved. This paper settles the square grid, both exactly and at constant total-variation error.

## Results

Let $U_L$ be the uniform distribution on domino tilings of the free $L\times L$ square ($L$ even), as a distribution on its $n=2L(L-1)$ edge indicators.

- **Profile bound (Theorems A and A′).** For every split of the edges into two parts of size at least $n/3$ and every vertex set $S$, the probability that the matching edges in the first part cover exactly $S$ is at most $e^{-c(L)L}$, with $c(L)\ge 1/1129$ for $L\ge1000$. The same holds for degree-balanced vertex cuts. The proof uses a tilted Kasteleyn determinant, Carleman's determinant inequality and an elementary free-square bound $\lvert K^{-1}(w,b)\rvert\le 4A/\lVert w-b\rVert_\infty$ with $A\approx1.37$.
- **Exact monotone and tropical circuits (Theorem D).** Monotone arithmetic circuits and $(\min,+)$ circuits for the square-grid matching polynomial need size $e^{\Omega(L)}$.
- **Probabilistic circuits (Theorem B).** A nonnegative, smooth, decomposable probabilistic circuit with leaves of arity at most $2n/3$ whose distribution is within total variation $\varepsilon<1$ of $U_L$ has size at least $(1-\varepsilon)^2e^{c(L)L}$. Determinism is not assumed.
- **Tightness (Theorem C).** A column-transfer circuit of size $O(L^2(1+\sqrt2)^L)$ computes $U_L$ exactly, so the optimal size is $e^{\Theta(\sqrt n)}$ for every fixed error below one.
- **Entanglement.** The Rokhsar–Kivelson dimer state on the square has Rényi-∞ entropy at least $c(L)L$ across every balanced bipartition of the links.

## What this paper does not claim

- A robust separation of the same kind on Valiant's triangular grid already follows from his 1980 argument plus standard steps. The paper states it first (Proposition 1.1). The new part is the square grid.
- $U_L$ has polynomial-size determinant-based generating circuits, and exact inference for it is easy. Known conversions of those circuits to signed probabilistic circuits are not syntactically decomposable. Whether small signed decomposable circuits exist for $U_L$ is open, except in the structured exact case (Proposition 9.1).
- The explicit constant is crude: about 100 times smaller than what the numerics show, and nothing explicit is proved for $L<264$.

## Verify

The checks in `verify/` test identities and constants and show how loose the bounds are. None of the proofs depends on them. From the repository root, with `uv` installed:

```sh
uv run python verify/kasteleyn_pgc_checks.py
uv run --with numpy --with scipy python verify/kernel_bounds.py
uv run python verify/constants.py
uv run --with numpy --with scipy python verify/avoidance_checks.py
uv run python verify/transfer_pc.py
python3 verify/lemma8_flux.py
```

Each prints JSON to standard output. The saved output of each run is next to its script. Appendix A of the paper says what each one checks.

To compile the paper with Tectonic:

```sh
cd paper
tectonic square-grid-dimers.tex
```

## Cite

See `CITATION.cff`.

## License

Paper text: CC BY 4.0. Code: MIT (see `LICENSE`).
