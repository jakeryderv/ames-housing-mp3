# Mini-Project 3: Predictive Modeling and Optimization for Real Estate Investment

## Problem Framing

A real estate investment company needs help making two connected decisions: how much each house is worth, and which houses to renovate under a fixed budget. Pure prediction isn't enough on its own (a price estimate doesn't tell you what to buy), and pure optimization isn't enough either (you can't allocate budget without knowing which properties will pay off).

The pipeline answers both:

1. **Neural network** trained on the Ames Housing dataset (2,930 properties, 82 features) to predict `SalePrice` from a small set of inputs.
2. **Linear program** (a 0/1 knapsack, technically) that uses those predictions to pick a subset of houses that maximizes profit under a fixed renovation budget.

This is a regression problem, not classification, because the target is a continuous dollar amount and the company cares about how much each house is worth, not which bucket it falls into.

## Feature Selection Justification

The project allows 6 to 8 input variables. We picked 8:

| Feature | Why |
|---|---|
| `Overall Qual` | Top-correlated feature in Ames (overall material and finish quality, correlation 0.80 with price) |
| `Gr Liv Area` | Above-grade living area, the main size driver |
| `Total Bsmt SF` | Basement size, extra usable space |
| `Garage Cars` | Garage capacity, a strong amenity signal |
| `Year Built` | Newer houses usually sell for more |
| `Year Remod/Add` | When the house was last renovated, directly relevant to Part 4 |
| `Neighborhood` | Location, location, location (one-hot encoded into 28 binary columns) |
| `Full Bath` | Bathroom count |

We initially picked `Lot Area` instead of `Year Remod/Add`, but swapped it out after checking the data: `Lot Area` is heavily right-skewed (skew 12.82) and only weakly correlated with price (0.27), while `Year Remod/Add` is much better behaved (skew -0.45, correlation 0.53) and ties cleanly into the renovation-budget framing of Part 4. Empirical val and test scores were within noise either way, so the swap was driven by methodology rather than raw accuracy.

## Data Preparation and Baseline

**Preprocessing:** dropped 2 rows with missing values (negligible loss out of 2,930), one-hot encoded `Neighborhood`, standard-scaled the numeric features. The target was also standard-scaled internally so the network trains on a well-behaved range.

**Splits:** 64% train, 16% val, 20% test (two nested 80/20 splits, fixed seed). The val set is for comparing architectures and tuning hyperparameters so we don't peek at test during selection. Test was kept for the final report.

**Metric:** RMSE in dollars. Same units as `SalePrice`, and it penalizes big misses (which matter for investment decisions). MAE and R^2 are reported alongside.

**Baseline:** predict the training mean for every test row. RMSE about \$90K, R^2 about 0.0. Any real model has to beat this.

## Model Comparison

We trained four neural network architectures with MLPRegressor (ReLU, Adam, early stopping):

| Model | Architecture | Val R^2 | Test R^2 |
|---|---|---|---|
| Baseline | mean-predictor | -0.00 | -0.00 |
| NN-A | (32,) | 0.873 | 0.867 |
| **NN-B** | **(64, 32)** | **0.876** | **0.887** |
| NN-C | (128,) | 0.863 | 0.890 |
| NN-D | (64, 32, 16) | 0.869 | 0.875 |

All four crush the baseline and cluster between R^2 0.86 and 0.89. Diminishing returns kick in fast at this dataset size.

**Key takeaways:**
- Depth slightly beats width: the wide single-layer net (NN-C) was the worst on val.
- Going past 2 layers didn't help: NN-D's third layer added complexity for no gain.
- Val and test scores match closely for every model, so no overfitting to either split.
- Gaps between top models are under 0.01 R^2, which is noise-range. Multiple random seeds would firm this up.

**Winner (chosen on val):** NN-B. Test R^2 0.887, RMSE about \$30K.

We then ran a 4 by 3 = 12-combo hyperparameter sweep on NN-B over `alpha` (L2 weight decay) and `learning_rate_init` (Adam step size). The best combo (`alpha=0.1, lr=0.01`) bumped val R^2 to 0.883, but test R^2 stayed flat at 0.884. The grid was mostly flat, confirming the model isn't very sensitive to these knobs. To meaningfully push past R^2 0.89 we'd need more features or a different model class (e.g. gradient boosting), not finer hyperparameter sweeps.

## Optimization Setup

The company has limited renovation capacity. The question: which houses should it pick to maximize expected investment gain?

The optimization runs on all 2,928 cleaned properties (not just the test set), using the tuned NN-B predictions applied to the full dataset. Two quantities are derived per property:

**Expected Gain** — the estimated market mispricing:

> ExpectedGain = PredictedPrice − SalePrice

A positive value means the NN thinks the property sold for less than it's worth, i.e., an undervalued opportunity. A negative value means it sold for more than the model expects.

**Renovation Burden** — a dimensionless proxy for the effort required to renovate each property, built from three Ames features:

> RenovationBurden = (10 − OverallCond) + (2026 − YearRemodAdd) / 10 + GrLivArea / 1000

The three terms capture condition (worse condition = more work), age since last remodel (older = more modernization needed), and size (larger = more labor). All components are grounded in real Ames features; no dollar costs are assumed.

The budget is derived statistically rather than set by hand: the company is assumed to handle 10% of the total renovation workload across all 2,928 properties, so BUDGET = 0.10 × Σ RenovationBurden.

The LP is a classic 0/1 knapsack:

- **Decision:** for each property, x_i ∈ {0, 1} — invest or skip
- **Objective:** maximize Σ ExpectedGain_i · x_i
- **Constraint:** Σ RenovationBurden_i · x_i ≤ BUDGET
- **Solver:** PuLP's CBC (open-source MILP solver), checked for Optimal status before results are trusted.

## Key Insights

**The optimizer targets market mispricing, not renovation ROI.** Unlike a renovation-simulation approach, ExpectedGain captures properties where the NN believes the market underpriced the house relative to its features — regardless of condition. The LP then filters those opportunities through a capacity constraint, selecting the highest-gain properties the company can actually handle.

**The burden constraint introduces a genuine trade-off.** High-gain properties often carry high renovation burden (poor condition, large area, or long-unmodeled age), so the optimizer cannot simply stack the top-gain properties. A single high-burden outlier can consume as much budget as several lower-burden properties, forcing the LP to weigh gain-per-burden-unit rather than raw gain.

**Selected properties tend to be well-maintained and recently remodeled.** Because RenovationBurden rises sharply with poor condition and age since last remodel, the constraint naturally favors properties with OverallCond ≥ 6 and recent remodel years — not because those are explicitly required, but because they leave more budget for additional picks. Large properties are penalized by the size term even if their condition is good.

**The budget is dataset-driven, not arbitrary.** Setting BUDGET = 10% × Σ RenovationBurden means the constraint scales with the actual composition of the market being analyzed. A dataset of older, larger, worse-condition homes produces a tighter effective budget than one of newer, smaller, well-maintained ones, which is the correct behavior for a capacity-limited firm.

## Limitations

- **Single random seed.** Every comparison (NN-A vs NN-B, the tuning grid, the log-transform check) used `random_state=42`. The val R^2 gaps between top models are under 0.01, which is in noise range. Running 3 to 5 seeds would tell us whether NN-B truly beats NN-A or just got lucky on this split.
- **ExpectedGain conflates prediction error with mispricing.** The quantity PredictedPrice − SalePrice captures both genuine undervaluation and NN prediction error. A property with a large positive gain might be a real bargain, or the NN might simply be wrong about it. With ~$30K test RMSE, the uncertainty on individual gain estimates is substantial and isn't quantified.
- **Renovation Burden is a proxy without dollar grounding.** The burden formula (condition + age + size terms) captures the right intuitions but has no connection to actual construction cost. The 10% budget fraction is similarly a calibration choice. Sensitivity to these design decisions isn't tested in the current implementation.
- **ExpectedGain can be negative, and the LP handles this implicitly.** Properties where the NN predicts less than the sale price contribute negative gain if selected. The LP avoids these automatically when optimizing, but there's no explicit filter, so near-zero or slightly negative properties near the budget boundary could be selected in edge cases.
- **Hyperparameter search was narrow.** We tuned `alpha` and `learning_rate_init` but didn't touch `solver`, `activation`, or `batch_size`. Lbfgs in particular often outperforms Adam on small datasets like this one.

## Conclusion

The pipeline answers the original business question end to end. The neural network turns 8 features into a roughly \$30K-RMSE price estimate (test R^2 about 0.89), well above the mean-predictor baseline. The linear program then uses those predictions to identify undervalued properties and selects the optimal portfolio under a renovation capacity constraint.

The prediction step is robust: NN-B beat 3 alternative architectures and held up across 12 hyperparameter combos, all chosen on a held-out val set. The optimization step is mathematically correct (LP returns Optimal status, constraint honored), but its interpretation rests on two design choices: that ExpectedGain = PredictedPrice − SalePrice is a meaningful proxy for undervaluation, and that the RenovationBurden formula captures the right property characteristics. Both are defensible but neither is ground truth. The honest takeaway: prediction tells us what each house is worth relative to its features; optimization tells us which undervalued houses we can actually afford to take on given capacity limits. Tightening the burden formula with real construction-cost data is the main thing a real engagement would need.
