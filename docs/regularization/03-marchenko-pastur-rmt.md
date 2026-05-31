# 03 · Random Matrix Theory & Marchenko–Pastur Eigenvalue Clipping

**Goal:** decide which eigenvalues of an estimated correlation matrix are **real
structure (factors)** and which are **pure sampling noise**, then suppress the
noise ones. This is the most direct cure for the inflated leave-one-out $R^2$.

Prerequisites: [foundations](00-foundations.md) §2,5,6 (eigendecomposition,
sample covariance, conditioning) plus the RMT theory below.

---

## 1. The empirical spectral distribution

For a $p\times p$ symmetric matrix, its **empirical spectral distribution (ESD)** is the histogram of its $p$ eigenvalues. RMT asks: *if the data were pure noise, what would that histogram look like?* — giving a baseline to test real eigenvalues against.

## 2. Prerequisite: Wishart matrices and the noise model

Let $X\in\mathbb{R}^{n\times p}$ have i.i.d. entries with mean 0, variance 1 (e.g. independent standardized series — **no real correlation**). The sample covariance $S=\tfrac1nX^\top X$ is a (white) **Wishart matrix**. Its true population covariance is $I$ (all true eigenvalues $=1$), so a "perfect" estimator would give every eigenvalue $=1$. RMT tells us how far from 1 the *sample* eigenvalues spread purely from finite $n$.

## 3. The Marchenko–Pastur law

As $p,n\to\infty$ with aspect ratio $q=p/n$ fixed, the ESD of the white Wishart $S$ converges to the **Marchenko–Pastur distribution** with density
$$ f_{\text{MP}}(\lambda)=\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi\,\sigma^2 q\,\lambda},\qquad \lambda\in[\lambda_-,\lambda_+], $$
supported between the **edges**
$$ \boxed{\ \lambda_\pm=\sigma^2\big(1\pm\sqrt{q}\big)^2\ }\qquad(\sigma^2=1\text{ for standardized data}). $$
(For $q>1$ there is also a point mass at 0 of size $1-1/q$ — the singular directions.) The law is derived via the **Stieltjes transform** $m(z)=\int\frac{f(\lambda)}{\lambda-z}d\lambda$, which for MP satisfies a quadratic self-consistency equation; the branch points of its solution are $\lambda_\pm$. You don't need the derivation to use it — the operational content is the edge formula.

**The headline:** even with *zero* true correlation, sample eigenvalues fan out across $[\lambda_-,\lambda_+]$. Example $p=40,\ n=200,\ q=0.2$: noise band $\approx[(1-0.447)^2,(1+0.447)^2]=[0.31,\ 2.09]$. **Any eigenvalue inside that band is statistically indistinguishable from noise.**

## 4. Signal as outliers: the spiked model and the BBP transition

Real data = noise + a few genuine factors. The **spiked covariance model** (Johnstone) has population covariance $=I$ plus a few large "spike" eigenvalues. RMT result (**Baik–Ben Arous–Péché, "BBP transition"**): a population spike of size $s$ produces a *visible* outlier eigenvalue above $\lambda_+$ **only if** $s>1+\sqrt q$; below that detectability threshold the factor is buried in the bulk. When detectable, the sample outlier sits at $\approx (1+s)\big(1+\tfrac{q}{s}\big)$ (biased upward), and its eigenvector is only partially aligned with the true factor (an **angle**/inconsistency that worsens as $q$ grows). Takeaways:
- eigenvalues **above $\lambda_+$** = real factors (but their magnitudes and directions are noisy);
- the **bulk** $[\lambda_-,\lambda_+]$ = noise;
- weak factors below the BBP threshold are simply **undetectable** at this $q$ — more data ($\downarrow q$) is the only fix.

## 5. The clipping procedure (Laloux–Bouchaud–Potters)

1. Estimate $q=p/n$ (use the *effective* $n$). Optionally **fit** the bulk to estimate $\sigma^2<1$ (the top factors absorb variance, so the noise band scales below 1) — "fitted MP."
2. Eigendecompose $C=\sum_k\lambda_k q_kq_k^\top$.
3. **Clip:** keep eigenvalues $\lambda_k>\lambda_+$ unchanged (signal); **replace** every $\lambda_k\le\lambda_+$ with a single constant $=$ their average, chosen so that $\operatorname{tr}$ is preserved (keeps unit diagonal / total variance). Rebuild $C_{\text{clean}}$.

This is a **nonlinear** operation on the spectrum: it treats big and small eigenvalues differently (unlike linear shrinkage, doc 02, which scales all of them).

**Better variants:**
- **eigenvalue *shrinkage***: instead of flattening the bulk to a constant, shrink bulk eigenvalues smoothly toward the mean;
- **Rotationally-Invariant Estimator (RIE) / Ledoit–Péché optimal nonlinear shrinkage**: the asymptotically optimal map $\lambda_k\mapsto\hat\lambda_k$ applied eigenvalue-by-eigenvalue (derived from the Stieltjes transform of the observed spectrum). RIE **generalizes both** MP-clipping and linear LW shrinkage and is the current state of the art. Use it if you want one principled denoiser.

## 6. Why this fixes the inflated $R^2$ specifically

The precision matrix weights each mode by $1/\lambda$: $\;\Theta=\sum_k\lambda_k^{-1}q_kq_k^\top$. A **noise** eigenvalue $\lambda_k\approx0.05$ contributes $1/0.05=20$ to $\Theta$. Since the leave-one-out $R^2_i=1-1/\Theta_{ii}$ (foundations §7), those inverted noise eigenvalues drive $R^2\to1$. The big-cluster $R^2$ we saw wasn't signal — it was $\sim$70 noise eigenvalues being inverted. **Clipping floors the noise eigenvalues**, so $\Theta$ is dominated by the few real factors and the spurious $R^2$ collapses to its true (lower) value.

**Bonus diagnostic:** the **number of eigenvalues above $\lambda_+$** $=$ the number of detectable factors $=$ the cluster's **effective dimensionality** = how many genuinely independent bets exist. (A 79-leg cluster may have effective rank 1–3.)

## 7. In our pipeline

- After Higham, RMT-clip (or RIE) each cluster's correlation matrix before computing precision / $R^2$ / factors.
- Our $q$ is often near or above 1 (legs $\approx$ observations), so the noise band is **wide** — expect aggressive clipping and few surviving factors. That's the honest picture: most of the apparent multivariate structure is undersampled noise.
- $q>1$ → point mass at 0 → the raw matrix is singular; clipping/RIE (or shrinkage) is mandatory, not optional.
- Implementation: clipping is a few lines on top of `numpy.linalg.eigh`; RIE via the `pyRMT`/`scikit-rmt` libraries or a short Ledoit–Péché implementation.

**One-line summary:** Marchenko–Pastur gives the exact eigenvalue spectrum of *pure noise* at aspect ratio $q$; anything inside that band is noise to be flattened, anything above it is a real factor to keep — which de-noises the inverse and kills the dimensionality-inflated $R^2$.
