# 04 · Graphical Lasso (Sparse Precision Estimation)

**Goal:** estimate a **sparse precision matrix** $\Theta=\Sigma^{-1}$ directly. Its
zero pattern is the **conditional-independence graph** (which markets are
*directly* linked, vs. linked only through others), and from the same fit you
read off de-biased partial correlations and leave-one-out $R^2$.

Prerequisites: [foundations](00-foundations.md) §7 (precision matrix identities)
plus the Gaussian-graphical-model, MLE, and lasso theory below.

---

## 1. Prerequisite: the multivariate Gaussian and its precision matrix

For $X\sim\mathcal{N}(\mu,\Sigma)$ the density is
$$ p(x)\propto \exp\!\Big(-\tfrac12 (x-\mu)^\top\Theta\,(x-\mu)\Big),\qquad \Theta=\Sigma^{-1}. $$
The **precision** $\Theta$ controls the *interactions*: the exponent expands into a sum of pairwise terms $-\tfrac12\Theta_{ij}(x_i-\mu_i)(x_j-\mu_j)$. So $\Theta_{ij}$ is the strength of the **direct** coupling between $i$ and $j$.

## 2. Prerequisite: conditional independence & Gaussian graphical models

Two cornerstone facts for Gaussians (foundations §7):
- **$\Theta_{ij}=0 \iff X_i\perp X_j \mid X_{\text{rest}}$** (conditionally independent given everything else);
- **partial correlation** $\rho_{ij\cdot\text{rest}}=-\Theta_{ij}/\sqrt{\Theta_{ii}\Theta_{jj}}$.

Build a graph with a node per variable and an **edge $i\!-\!j$ iff $\Theta_{ij}\neq0$**. This is a **Gaussian graphical model** (a Markov random field): an absent edge means "no direct link — any apparent correlation is routed through other nodes." This is the precise statement of **direct vs. indirect dependence** (BTC–SOL correlate only via ETH ⇒ $\Theta_{\text{BTC,SOL}}\approx0$, no edge).

## 3. Why we can't just invert $S$

The Gaussian **MLE** of $\Theta$ maximizes the log-likelihood
$$ \ell(\Theta)=\log\det\Theta-\operatorname{tr}(S\Theta) \quad\Rightarrow\quad \hat\Theta_{\text{MLE}}=S^{-1}. $$
But at $q=p/n$ near or above 1, $S$ is ill-conditioned or singular, so $S^{-1}$ is noise and **dense** (every $\Theta_{ij}\neq0$ from sampling error — no graph structure). We need to *regularize* toward **sparsity**: most pairs of markets are not *directly* linked, so most $\Theta_{ij}$ should be exactly 0.

## 4. Prerequisite: the lasso and $\ell_1$ sparsity

The **lasso** adds an $\ell_1$ penalty to a fit: $\min \|y-X\beta\|^2+\lambda\|\beta\|_1$. Unlike the $\ell_2$ (ridge) penalty, the $\ell_1$ penalty has corners on the coordinate axes, so the solution lands *exactly on zero* for many coefficients — it performs **variable selection** (sparsity), not just shrinkage. Graphical lasso applies this idea to the entries of $\Theta$.

## 5. The Graphical Lasso objective (Friedman–Hastie–Tibshirani 2008)

Maximize the **penalized** Gaussian log-likelihood:
$$ \boxed{\ \hat\Theta=\arg\max_{\Theta\succ0}\ \log\det\Theta-\operatorname{tr}(S\Theta)-\lambda\|\Theta\|_1\ } $$
where $\|\Theta\|_1=\sum_{i\neq j}|\Theta_{ij}|$ penalizes off-diagonals.
- $\log\det\Theta-\operatorname{tr}(S\Theta)$ = Gaussian likelihood (maximized at $S^{-1}$);
- $\lambda\|\Theta\|_1$ = lasso penalty → drives small $\Theta_{ij}$ to **exactly 0** (a sparse graph);
- $\Theta\succ0$ = stays a valid precision matrix.
The objective is **convex** (concave in $\Theta$ over the PD cone), so there's a unique global optimum.

## 6. How it's solved, and the regression connection

Coordinate descent over columns: cycling one variable at a time, the optimality conditions for the block reduce to a **lasso regression of that variable on all the others** (the sub-problem is $\min_\beta \tfrac12\beta^\top W_{11}\beta - \beta^\top s_{12}+\lambda\|\beta\|_1$). This is the beautiful link:

> **Graphical lasso ≈ "regress each node on all the others with an $\ell_1$ penalty," stitched together into one PD-consistent matrix.**

This is exactly the **neighborhood-selection** view of **Meinshausen–Bühlmann (2006)** (run a separate lasso per node to find its graph neighbors); graphical lasso is the joint, likelihood-based version that guarantees a single coherent PSD $\Theta$. It also makes clear why glasso *directly produces* the leave-one-out $R^2$: each node's regression-on-the-rest is built in.

## 7. Choosing $\lambda$ (model selection)

$\lambda$ dials from dense/noisy ($\lambda=0\Rightarrow S^{-1}$) to very sparse/biased (large $\lambda$). Pick it by:
- **cross-validated** held-out Gaussian log-likelihood (maximize predictive likelihood); or
- **eBIC** (extended BIC) — BIC with an extra term penalizing the number of edges, designed for high-dimensional graph selection where ordinary BIC under-penalizes. eBIC is usually preferred when the goal is the *graph* rather than prediction.

## 8. The two-for-one payoff

A single regularized fit yields simultaneously:
1. the **sparse conditional-independence graph** — your true *direct*-dependency network (the irreducible skeleton, with transitive/indirect edges removed); and
2. a **stable, de-biased precision matrix**, from which $\rho_{ij\cdot\text{rest}}$ and $R^2_i=1-1/(\Sigma_{ii}\Theta_{ii})$ follow directly — with the **dimensionality inflation gone**, because the noise off-diagonals are now exactly zero rather than small-noisy.

It thus **replaces both** the naïve inversion (doc's earlier ad-hoc precision) *and* the separate per-leg ridge regressions, in one estimate.

## 9. Relationship to the other techniques

- vs **shrinkage (02) / RMT (03):** those regularize the *covariance* (toward a target / by flattening noise eigenvalues). Graphical lasso regularizes the *inverse* toward **sparsity** (graph structure). They compose: Higham → (RMT/shrinkage to denoise $S$) → graphical lasso on the cleaned $S$ for the sparse graph. RMT controls the spectrum; glasso controls the support — different, complementary knobs.
- Both glasso and RMT "denoise the inverse," but RMT does it by eigenvalues, glasso by entrywise sparsity.

## 10. In our pipeline

- Input: the Higham-repaired (and ideally RMT-denoised) per-cluster correlation matrix from Hayashi–Yoshida.
- Output: a partial-correlation graph showing which market links are *direct* — e.g. within the Iran complex, perhaps "nuclear deal" links directly to "sanctions relief" and "enrichment," and the other legs hang off those indirectly → hedge through the hubs, treat the peripherals as redundant. This graph is also the clean input to signed structural-balance (frustrated triangles = mispricings).
- **Caveat:** glasso assumes (sub-)Gaussianity. Prediction-market prices live in $[0,1]$; transform to **logit / log-odds** first so the Gaussian model — and hence the conditional-independence interpretation — is appropriate.
- Implementation: `sklearn.covariance.GraphicalLassoCV` (CV-tuned $\lambda$); for eBIC and speed, `skggm` (QUIC).

**One-line summary:** graphical lasso = maximum-likelihood precision estimation with an $\ell_1$ penalty, giving a sparse conditional-independence graph (direct vs. indirect links) and a regularized, de-biased precision matrix in one convex fit.
