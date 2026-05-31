# 02 · Ledoit–Wolf / OAS Shrinkage

**Goal:** turn the noisy, ill-conditioned sample covariance $S$ into a stable,
well-conditioned estimate by pulling it toward a structured **target** $F$, with
the mixing weight chosen *optimally and automatically* from the data.

Prerequisites: [foundations](00-foundations.md) §5–6 (sample covariance,
bias–variance/MSE, conditioning) plus the shrinkage theory below.

---

## 1. The problem: $S$ is unbiased but high-variance

$S=\tfrac1nX^\top X$ is unbiased ($\mathbb{E}S=\Sigma$) but when $q=p/n$ is not small its **eigenvalues spread**: a known consequence of sampling is that the largest sample eigenvalues are biased **upward** and the smallest **downward** relative to the true ones (this is exactly the Marchenko–Pastur spreading, doc 03). So:
- $\kappa(S)=\lambda_{\max}/\lambda_{\min}$ is far too large → $S^{-1}$ is unstable;
- if $p>n$, $S$ is singular and $S^{-1}$ doesn't exist at all.
Either way, anything using $S^{-1}$ (precision, partial correlation, $R^2$, optimal portfolio weights) is dominated by noise.

## 2. Prerequisite theory: shrinkage and Stein's paradox

The deep idea is **Stein's paradox (1956) / the James–Stein estimator**: when estimating many parameters at once, the "obvious" unbiased estimator (here $S$) is *inadmissible* — you can strictly lower total risk (MSE) by **shrinking** every estimate toward a common point. You accept some bias to kill a lot of variance. Shrinkage covariance estimation (Ledoit–Wolf 2004) is the matrix version.

**The estimator.** Convex-combine $S$ with a low-variance target $F$:
$$ \hat\Sigma(\delta) = (1-\delta)\,S + \delta\,F,\qquad \delta\in[0,1]. $$
- $S$: unbiased, high variance.
- $F$: biased (wrong in general) but very low variance (few parameters).
- The blend trades a controlled amount of bias for a large variance reduction. It **always improves conditioning**: eigenvalues of $\hat\Sigma(\delta)$ are $(1-\delta)\lambda_k(S)+\delta\lambda_k(F)$, pulled away from 0 toward the target's spectrum, so $\kappa$ drops and the matrix becomes invertible even when $S$ wasn't.

This is also exactly **ridge-style / Bayesian** regularization: shrinking toward $\mu I$ is a Gaussian conjugate-prior posterior mean with the prior centered on a spherical covariance.

## 3. Choosing the target $F$ (the prior you impose)

$F$ encodes your structural belief; common choices, increasing in structure:

| target | form | belief |
|---|---|---|
| scaled identity | $F=\mu I,\ \mu=\tfrac{1}{p}\operatorname{tr}S$ | no correlation, equal variances (Ledoit–Wolf 2004) |
| constant correlation | same variances as $S$, every $\rho_{ij}=\bar\rho$ | one common correlation level — natural for a co-moving cluster (Ledoit–Wolf 2003) |
| single factor | $F$ from a one-factor market model | one dominant driver (a "beta") |

For a basket of markets that genuinely co-move, **constant-correlation** is the standard, well-motivated target.

## 4. The key result: the optimal $\delta$ in closed form

Ledoit–Wolf choose $\delta$ to minimize the expected loss
$$ \delta^\star=\arg\min_\delta\ \mathbb{E}\,\big\|\hat\Sigma(\delta)-\Sigma\big\|_F^2. $$
Expanding the quadratic and using $\mathbb{E}S=\Sigma$ gives a clean bias–variance decomposition whose minimizer is
$$ \boxed{\ \delta^\star=\frac{\sum_{ij}\operatorname{Var}(S_{ij})}{\sum_{ij}\big(\mathbb{E}[S_{ij}]-F_{ij}\big)^2}\ }\;\approx\;\frac{\text{estimation noise in }S}{\text{distance from }S\text{ to the target}}. $$
Intuition:
- **numerator** = how noisy $S$ is (grows with $p/n$) → noisier data ⇒ shrink **more**;
- **denominator** = how far the target is from the truth ($=$ how much bias shrinking introduces) → if the target is badly wrong, shrink **less**.
Crucially, both numerator and denominator are **estimable from the sample alone** — Ledoit–Wolf give consistent plug-in estimators (the numerator from the dispersion of the per-observation outer products $x_kx_k^\top$ around $S$). So $\delta^\star$ is computed directly, **no cross-validation**, and it is asymptotically optimal as $p,n\to\infty$ with $p/n$ fixed.

## 5. OAS — the small-sample refinement

**Oracle Approximating Shrinkage** (Chen, Wiesel, Hero 2010) targets the *oracle* shrinkage (the unknowable $\delta$ that would be optimal if $\Sigma$ were known) and, under (sub-)Gaussian data, converges to it faster than Ledoit–Wolf when $n$ is **small** — our regime. For the scaled-identity target it has the closed form
$$ \delta_{\text{OAS}}=\min\!\left(1,\ \frac{\big(1-\tfrac2p\big)\operatorname{tr}(S^2)+\operatorname{tr}(S)^2}{\big(n+1-\tfrac2p\big)\big(\operatorname{tr}(S^2)-\tfrac1p\operatorname{tr}(S)^2\big)}\right). $$
Use **OAS** when $n$ is small and data is roughly Gaussian; use **LW with the constant-correlation target** when you specifically want the "one common correlation" prior.

## 6. Relationship to the other techniques

- Shrinkage applies **one linear** map to the whole spectrum: $\lambda_k\mapsto(1-\delta)\lambda_k+\delta\,\lambda_k(F)$ — it pulls *every* eigenvalue toward the target, including the **true factor** eigenvalues, which it therefore biases downward. That's its weakness.
- **RMT / nonlinear shrinkage (doc 03)** fixes that by shrinking eigenvalues *nonlinearly* — leave the big (signal) ones, crush only the small (noise) ones. Ledoit–Péché's optimal nonlinear shrinkage (RIE) is the generalization that dominates linear LW.
- vs **Higham (doc 01):** Higham fixes *validity* (PSD); shrinkage fixes *variance/conditioning*. Shrinking toward a PD target with $\delta>0$ usually also restores PD as a side effect, but do Higham first to be safe.

## 7. In our pipeline

- Replaces my hand-set $\delta=0.10$ with a per-cluster, MSE-optimal $\delta^\star$ derived from that cluster's own $n,p$ and correlation dispersion.
- Caveat: the LW/OAS derivation assumes roughly i.i.d. samples. With **overlapping** Hayashi–Yoshida intervals the *effective* $n$ is smaller than the raw observation count, so the formula slightly **under-shrinks** — treat $\delta^\star$ as a floor and lean toward more shrinkage.
- Implementation: `sklearn.covariance.LedoitWolf`, `sklearn.covariance.OAS` (both return the shrunk matrix and the chosen $\delta$).

**One-line summary:** shrinkage = James–Stein for covariance — accept a little bias by mixing in a structured target, with a closed-form data-driven weight that shrinks harder exactly when the data is noisier.
