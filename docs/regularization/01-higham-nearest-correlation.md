# 01 · Higham Nearest Correlation Matrix

**Goal:** given an estimated matrix $C_0$ that is *supposed* to be a correlation
matrix but isn't quite valid (not PSD), find the **closest genuine correlation
matrix** $X$. This is the step that repairs the matrix so everything downstream
(inversion, factor analysis, shrinkage) is operating on a real object.

Prereqsuites: [foundations](00-foundations.md) §2–4 (eigendecomposition, PSD,
correlation matrices) plus the convex-projection theory built up below.

---

## 1. Why $C_0$ is broken (recap)

A valid correlation matrix lives in the **elliptope** $\mathcal{E}$ = {symmetric, PSD, unit diagonal}. When you assemble $C_0$ from **pairwise** estimates — each $\rho_{ij}$ from its own data (Hayashi–Yoshida uses each pair's own overlapping observation times; "pairwise-complete" deletion does the same) — there is no single data matrix $X$ that all entries are the Gram matrix of, so the joint constraints (the correlation triangle inequality, foundations §4) are generically violated. Result: $C_0\notin\mathcal{E}$, with one or more **negative eigenvalues**. Inverting it, Cholesky, or calling it a covariance are then all invalid. We need to project $C_0$ back onto $\mathcal{E}$.

## 2. Prerequisite theory: convex projection

**Convex set.** $\mathcal{K}$ is convex if the segment between any two of its points stays in $\mathcal{K}$. Both ingredients here are convex:
- $\mathcal{S}_+$ = PSD matrices (a convex cone, foundations §3);
- $\mathcal{U}$ = symmetric matrices with unit diagonal (an **affine** set: fix the diagonal, off-diagonals free).
The elliptope is their intersection $\mathcal{E}=\mathcal{S}_+\cap\mathcal{U}$, also convex.

**Projection.** For a closed convex $\mathcal{K}$ and a point $C_0$, the projection
$$ P_\mathcal{K}(C_0)=\arg\min_{X\in\mathcal{K}}\|X-C_0\|_F^2 $$
**exists and is unique** (the defining property of closed convex sets in a Hilbert space). So "the nearest correlation matrix" is well-posed:
$$ X^\star=\arg\min_{X\in\mathcal{E}}\|X-C_0\|_F^2 = P_{\mathcal{E}}(C_0). $$
Higham (2002) introduced this formulation and the algorithm below; a weighted Frobenius norm $\|A\|_W=\|W^{1/2}AW^{1/2}\|_F$ can prioritize trustworthy entries.

## 3. The two easy projections

We can't project onto $\mathcal{E}$ directly, but we can onto each piece.

**Projection onto the PSD cone $\mathcal{S}_+$.** This is the key lemma. Eigendecompose the symmetric part $A=Q\Lambda Q^\top$. Then
$$ P_{\mathcal{S}_+}(A) = Q\Lambda_+Q^\top, \qquad \Lambda_+=\operatorname{diag}\big(\max(\lambda_k,0)\big). $$
**Derivation:** minimizing $\|X-A\|_F^2$ over $X\succeq0$. Because the Frobenius norm is orthogonally invariant, work in $A$'s eigenbasis; the problem separates into choosing eigenvalues $\mu_k\ge0$ minimizing $\sum_k(\mu_k-\lambda_k)^2$, whose solution is $\mu_k=\max(\lambda_k,0)$. So **"clip the negative eigenvalues to zero"** is exactly nearest-PSD. *(This single step is what my ad-hoc fix did.)*

**Projection onto the unit-diagonal set $\mathcal{U}$.** Just overwrite the diagonal:
$$ P_{\mathcal{U}}(A)=A-\operatorname{diag}(A)+I \quad(\text{set every } A_{ii}=1). $$

## 4. Why you can't just alternate them

Clip to PSD → the diagonal drifts off 1 (eigenvalue surgery changes the diagonal). Reset the diagonal to 1 → that perturbation re-introduces a negative eigenvalue. Naively alternating $P_{\mathcal{S}_+}$ and $P_{\mathcal{U}}$ (the **method of alternating projections / POCS**, von Neumann) converges to *some* point in $\mathcal{S}_+\cap\mathcal{U}$, but **not the nearest one** — it overshoots, because each projection forgets the displacement the other one needs.

## 5. Dykstra's correction — the fix

**Dykstra's algorithm** is alternating projection with a memory term that removes the overshoot, and it provably converges to the true projection onto the *intersection*. Maintain correction matrices $\Delta_S,\Delta_U$ (initially 0). Iterate:
$$
\begin{aligned}
R &= Y_k - \Delta_S, & X_{k+1} &= P_{\mathcal{S}_+}(R), & \Delta_S &\leftarrow X_{k+1}-R,\\
R' &= X_{k+1} - \Delta_U, & Y_{k+1} &= P_{\mathcal{U}}(R'), & \Delta_U &\leftarrow Y_{k+1}-R'.
\end{aligned}
$$
The corrections $\Delta$ "add back" each step the displacement the *other* constraint had imposed, so the iterates approach $P_{\mathcal{E}}(C_0)$ rather than a generic feasible point. Converges linearly; a few dozen iterations suffice for our matrix sizes. (Newton-based solvers, Qi–Sun 2006, are faster for large $p$ but Dykstra is the canonical, simple version.)

## 6. Algorithm summary

```
input:  C0 (symmetric, ~unit diagonal, possibly indefinite)
ΔS = 0;  Y = C0
repeat:
    R  = Y − ΔS
    X  = eig-clip(R)            # P_{S+}: clip negative eigenvalues to 0
    ΔS = X − R
    Y  = X with diagonal reset to 1   # P_U
until ‖Y − X‖_F small
return Y   # nearest correlation matrix
```

## 7. Why this beats the one-shot eigenvalue clip

A single $P_{\mathcal{S}_+}$ followed by a diagonal renormalization ($D^{-1/2}XD^{-1/2}$) is **not** $P_{\mathcal{E}}$:
- it can leave (or re-introduce) tiny negative eigenvalues after the rescale;
- it is *not the closest* valid matrix — it distorts off-diagonal correlations more than necessary.
Higham returns the **minimum-distortion** valid correlation matrix, so the partial correlations, eigenvalues, and $R^2$ you compute afterward are as faithful as possible to your original (HY) estimates. Garbage-minimization, not garbage-in/garbage-out.

## 8. In our pipeline

- Input: the per-cluster Hayashi–Yoshida correlation matrix (pairwise-assembled, almost always indefinite for clusters with $p\ge3$, signed edges, and sparse async overlaps).
- Use a **weighted** Higham if you want HY pairs with more overlapping observations to be perturbed less (weight $\propto$ overlap count).
- Output feeds shrinkage / RMT / graphical lasso, all of which assume a valid PSD correlation matrix.
- Implementation: `statsmodels.stats.correlation_tools.corr_nearest` (true Higham) vs `corr_clipped` (the cheap one-shot clip).

**One-line summary:** Higham = "find the nearest set of numbers that really *are* the cosines of one consistent set of unit vectors," via Dykstra-corrected alternating projection onto the PSD cone and the unit-diagonal set.
