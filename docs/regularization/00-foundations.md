# 00 · Foundations

Shared prerequisite theory for the four covariance-regularization techniques
(Higham, Ledoit–Wolf, Marchenko–Pastur, Graphical Lasso). Each technique doc
assumes this file.

---

## 1. Vectors, matrices, inner products, norms

- A real **symmetric** matrix $A \in \mathbb{R}^{p\times p}$ satisfies $A = A^\top$. All covariance and correlation matrices are symmetric.
- The **Frobenius inner product** of two matrices is $\langle A,B\rangle_F = \operatorname{tr}(A^\top B) = \sum_{ij}A_{ij}B_{ij}$, and the **Frobenius norm** is $\|A\|_F = \sqrt{\sum_{ij}A_{ij}^2}$. This is just the Euclidean norm on the $p^2$ entries; "nearest matrix" problems use it because it makes the geometry Euclidean.

## 2. Eigenvalues and the spectral theorem

For any real symmetric $A$, the **spectral theorem** gives an orthonormal eigenbasis:
$$ A = Q\Lambda Q^\top = \sum_{k=1}^p \lambda_k\, q_k q_k^\top, \qquad Q^\top Q = I,\ \ \Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_p). $$
- The $\lambda_k$ are **real**; the $q_k$ (columns of $Q$) are orthonormal eigenvectors.
- Each term $\lambda_k q_k q_k^\top$ is a rank-1 "mode." Reading a covariance this way: $q_k$ is a **portfolio** (linear combination of variables) and $\lambda_k$ is that portfolio's **variance**.
- Frobenius norm in terms of eigenvalues: $\|A\|_F^2 = \sum_k \lambda_k^2$. Trace: $\operatorname{tr}(A)=\sum_k\lambda_k$ (= total variance for a covariance).
- **Functions of a symmetric matrix** act on eigenvalues: $f(A) = Q\,f(\Lambda)\,Q^\top$. E.g. the inverse $A^{-1}=\sum_k \lambda_k^{-1} q_kq_k^\top$ — note the $1/\lambda_k$, which is why tiny eigenvalues make the inverse explode.

## 3. Positive (semi)definiteness — the central concept

A symmetric $A$ is **positive semidefinite (PSD)**, written $A\succeq 0$, if any of these *equivalent* conditions hold:

1. $w^\top A w \ge 0$ for every vector $w$ (no portfolio has negative variance);
2. all eigenvalues $\lambda_k \ge 0$;
3. $A = V^\top V$ for some matrix $V$ (a **Gram matrix** of the columns of $V$);
4. $A$ has a Cholesky factorization $A = LL^\top$.

**Positive definite (PD)** $A\succ 0$ strengthens these to strict inequality ($w^\top A w>0$, $\lambda_k>0$), which is exactly when $A$ is invertible *and* well-behaved.

Why this is "the" property: a matrix is the covariance of *some* real random vector **iff** it is symmetric PSD. So PSD is the dividing line between "a real statistical object" and "a meaningless table of numbers."

- The set of $p\times p$ PSD matrices is a closed **convex cone** $\mathcal{S}_+$ (closed under non-negative combinations; if $A,B\succeq0$ then $\alpha A+\beta B\succeq0$ for $\alpha,\beta\ge0$). Convexity is what makes projection onto it (Higham) well-defined.

## 4. Covariance and correlation matrices

For a random vector $X=(X_1,\dots,X_p)$:
- **Covariance** $\Sigma_{ij} = \operatorname{Cov}(X_i,X_j)=\mathbb{E}[(X_i-\mu_i)(X_j-\mu_j)]$; $\Sigma\succeq0$ always.
- **Correlation** $C_{ij}=\Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}=\rho_{ij}$; this is $\Sigma$ rescaled to unit diagonal, $C = D^{-1/2}\Sigma D^{-1/2}$ with $D=\operatorname{diag}(\Sigma)$. So $C$ is PSD with $C_{ii}=1$.

**Correlations are cosines.** Treat each (de-meaned, unit-scaled) variable as a vector; then $\rho_{ij}=\cos\theta_{ij}$ where $\theta_{ij}$ is the angle between variables $i$ and $j$. A valid correlation matrix = a Gram matrix of **unit vectors**. This is the geometric reason for the constraints below.

- The set of valid correlation matrices (symmetric, PSD, unit diagonal) is a convex body called the **elliptope**. Its surface is curved, so independently chosen pairwise $\rho$'s generically land *outside* it for $p\ge3$.
- **Correlation triangle inequality** (the $p=3$ slice of PSD):
$$ \rho_{13}\in\Big[\rho_{12}\rho_{23}-\sqrt{(1-\rho_{12}^2)(1-\rho_{23}^2)},\ \ \rho_{12}\rho_{23}+\sqrt{(1-\rho_{12}^2)(1-\rho_{23}^2)}\Big]. $$
Violate it and the $3\times3$ matrix has a negative eigenvalue. (Angle version: $\theta_{13}\le\theta_{12}+\theta_{23}$.)

## 5. Estimation: sample covariance, bias, variance, MSE

Given $n$ observations stacked as $X\in\mathbb{R}^{n\times p}$ (columns de-meaned), the **sample covariance** is
$$ S = \tfrac1n X^\top X \quad(\text{or }\tfrac1{n-1}). $$
- $S = \tfrac1n X^\top X$ is a Gram matrix → **PSD by construction**. (This matters: PSD failures arise only when entries are *not* all computed from one common $X$ — see Higham doc.)
- $S$ is **unbiased** ($\mathbb{E}[S]=\Sigma$) but **high variance** when $n$ is not $\gg p$. Quality is measured by **mean-squared error**
$$ \operatorname{MSE}(\hat\Sigma)=\mathbb{E}\,\|\hat\Sigma-\Sigma\|_F^2=\underbrace{\|\mathbb{E}\hat\Sigma-\Sigma\|_F^2}_{\text{bias}^2}+\underbrace{\mathbb{E}\|\hat\Sigma-\mathbb{E}\hat\Sigma\|_F^2}_{\text{variance}}. $$
The **bias–variance tradeoff**: $S$ has zero bias but large variance; a biased-but-stable estimator can have lower total MSE. This is the entire justification for shrinkage (Ledoit–Wolf) and regularization in general.

- **Aspect ratio** $q=p/n$ controls difficulty. $q\to0$: $S$ is excellent. $q\approx1$: $S$ is noisy and near-singular. $q>1$ ($p>n$): $S$ is **singular** (rank $\le n$), so $S^{-1}$ doesn't exist. Our market clusters routinely have $q$ near or above 1.

## 6. Conditioning and why the inverse is dangerous

- **Condition number** $\kappa(A)=\lambda_{\max}/\lambda_{\min}$. Large $\kappa$ ⇒ inverting $A$ amplifies noise by up to $\kappa$.
- Because $A^{-1}=\sum_k\lambda_k^{-1}q_kq_k^\top$, a small noisy eigenvalue $\lambda_k\approx\varepsilon$ contributes $1/\varepsilon$ to the inverse. Sampling noise pushes the *smallest* eigenvalues of $S$ toward 0 (eigenvalue "spreading"), so $S^{-1}$ is dominated by noise. Every downstream object that needs an inverse — **precision matrix, partial correlations, Mahalanobis distance, regression/$R^2$, portfolio weights** — inherits this instability. Regularization = controlling those small eigenvalues.

## 7. The precision matrix (used heavily in the Graphical Lasso doc)

$\Theta=\Sigma^{-1}$ is the **precision** (concentration) matrix. For a multivariate Gaussian it encodes *conditional* structure:
- $\Theta_{ij}=0 \iff X_i \perp X_j \mid X_{\text{rest}}$ (conditional independence);
- **partial correlation** $\rho_{ij\cdot\text{rest}}=-\Theta_{ij}/\sqrt{\Theta_{ii}\Theta_{jj}}$;
- **leave-one-out $R^2$** of regressing $X_i$ on all other variables: $R^2_i = 1-\dfrac{1}{\Sigma_{ii}\,\Theta_{ii}}$; for a correlation matrix ($\Sigma_{ii}=1$) this is $R^2_i=1-1/\Theta_{ii}$.

These three identities are why "get a good, invertible $\Sigma$" is the whole game.

---

### Reading order
1. [01 · Higham nearest correlation matrix](01-higham-nearest-correlation.md) — restore **validity** (PSD).
2. [02 · Ledoit–Wolf / OAS shrinkage](02-ledoit-wolf-shrinkage.md) — reduce **variance / conditioning**.
3. [03 · Marchenko–Pastur / RMT](03-marchenko-pastur-rmt.md) — separate **signal from noise** eigenvalues.
4. [04 · Graphical Lasso](04-graphical-lasso.md) — estimate a **sparse precision** matrix / dependency graph.
