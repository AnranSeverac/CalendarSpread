# Covariance Regularization — Theory Notes

Prerequisite theory for the four techniques used to extract reliable structure
from the (short, asynchronous, $p\approx n$) correlation matrices of the
prediction-market clusters. Written to be read in order.

## Why any of this is needed

Everything we want — partial correlations, leave-one-out $R^2$, factor structure,
the dependency graph, dispersion signals — requires **inverting an estimated
correlation matrix**. But the matrix we have is (a) not even a valid correlation
matrix (the pairwise Hayashi–Yoshida assembly isn't PSD) and (b) estimated from
too few, too-noisy observations, so its inverse is dominated by noise. These four
techniques fix different parts of that, and they **compose in a pipeline**.

## The pipeline

| step | technique | fixes | doc |
|---|---|---|---|
| 0 | (prerequisites) | linear algebra, PSD, estimation, precision identities | [00](00-foundations.md) |
| 1 | **Higham nearest correlation matrix** | **validity** — projects the indefinite HY matrix onto the nearest valid (PSD, unit-diagonal) correlation matrix | [01](01-higham-nearest-correlation.md) |
| 2 | **Ledoit–Wolf / OAS shrinkage** *or* **Marchenko–Pastur / RIE** | **conditioning / noise** — shrink the spectrum (LW, one linear pull) or denoise it (RMT/RIE, nonlinear: keep factors, crush noise) | [02](02-ledoit-wolf-shrinkage.md), [03](03-marchenko-pastur-rmt.md) |
| 3 | **Graphical Lasso** | **the inverse + structure** — sparse precision → conditional-independence graph (direct vs. indirect links) + de-biased $R^2$ | [04](04-graphical-lasso.md) |

Higham is non-negotiable as step 1 (everything downstream assumes a real matrix).
Step 2 is the big fix for the inflated $R^2$ (it removes the noise eigenvalues that
$1/\lambda$ inversion was amplifying). Step 3 turns the cleaned matrix into the
sparse dependency graph and the de-biased spanning numbers.

## How they differ (they all "regularize," but differently)

- **Higham** — operates on *validity* (project onto the elliptope). Minimum-distortion repair, no statistical assumption.
- **Ledoit–Wolf / OAS** — operates on the *whole spectrum linearly* (one shrink intensity $\delta^\star$ toward a structured target); biases even the true factors.
- **Marchenko–Pastur / RIE** — operates on the *spectrum nonlinearly* (leave signal eigenvalues, flatten the MP noise bulk); the optimal single denoiser, and it also reports the **number of real factors**.
- **Graphical Lasso** — operates on the *inverse's support* ($\ell_1$ sparsity), giving the conditional-independence graph and partial correlations.

## How it connects back to the trade

- The de-biased **LOO $R^2$** (steps 1–3) tells you how *spanned* each market is by its clustermates — high $R^2$ ⇒ predictable ⇒ deviations from the fit are tradeable.
- The **partial-correlation graph** (step 3) shows the *direct* links — hedge through the hubs, ignore redundant peripherals.
- The **factor count** (step 3 of doc 03) = effective dimensionality = how many independent bets a cluster really contains.
- These feed the downstream graph-theory layer (MST, communities, k-core, signed structural balance) and the factor-residual dispersion signal.

## Honest caveat for our data

With $q=p/n$ near or above 1 and only ~29 days of async history, the noise band is
wide and the detectable-factor threshold (BBP, doc 03) is high — so expect
aggressive denoising and few surviving factors. The regularization makes the
numbers *honest*, but the real ceiling on signal is **data quantity**; deeper
(daily, forward-captured) history does more than any estimator refinement.
