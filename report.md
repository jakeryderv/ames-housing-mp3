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

The company has a fixed renovation budget. The question: which houses should it pick to maximize profit?

For each of the 586 test-set houses, we simulated a renovation by modifying the features, then asked the NN twice (before and after):

- `Overall Qual` + 1 (capped at 10)
- `Gr Liv Area` + 200 sqft
- `Year Remod/Add` set to 2010 (the renovation just happened)

Then per house:

- **Current value** = NN prediction on original features
- **Post-reno value** = NN prediction on modified features
- **Uplift** = post-reno minus current
- **Reno cost** = 10% of current value (a proxy: bigger houses cost more to renovate)
- **Profit** = uplift minus reno cost

The LP is a classic 0/1 knapsack:

- **Decision:** for each house, x_i is either 0 (skip) or 1 (renovate)
- **Objective:** maximize total profit across selected houses
- **Constraint:** total renovation cost stays at or below the budget B
- **Solver:** PuLP's CBC (an open-source MILP solver). The code asserts the solver returns "Optimal" status and that the budget constraint is honored before trusting the output.

With B = \$200K, the LP picks **17 houses** for **\$693K total profit** (about \$41K each), spending \$199K of the \$200K budget.

## Key Insights

**The picks cluster in affordable neighborhoods** (NAmes, OldTown, IDOTRR, Blueste, BrkSide, SWISU, Edwards, NWAmes, Crawfor). With reno cost = 10% of value, a \$400K house costs \$40K to renovate but the NN's predicted uplift is bounded, so the ROI is bad. A \$100K house with similar uplift gives roughly 4x the profit per dollar. The budget rewards profit-per-dollar, not absolute profit.

**The cost model matters more than the budget.** A sensitivity check with a flat \$30K reno cost (200 sqft at \$50/sqft labor plus \$20K quality upgrade, a more "construction-cost" view) produces a very different answer: only 6 houses picked, more expensive ones (avg \$187K vs \$117K), \$286K total profit instead of \$693K. Same data, same NN, same LP, but a different cost assumption flips the qualitative answer.

**The profit-vs-budget curve is concave** under either cost model: doubling the budget less than doubles the profit. Once the highest-ROI properties are picked, marginal additions earn less. Useful for the company to know where additional capital stops paying off.

## Limitations

- **Single random seed.** Every comparison (NN-A vs NN-B, the tuning grid, the log-transform check) used `random_state=42`. The val R^2 gaps between top models are under 0.01, which is in noise range. Running 3 to 5 seeds would tell us whether NN-B truly beats NN-A or just got lucky on this split.
- **Cost model is a proxy.** The 10%-of-value rule isn't grounded in real construction pricing. Sensitivity testing in the notebook shows it's the main lever driving the "cheap houses win" finding. A real engagement would tighten this with construction cost data.
- **NN extrapolation on renovated features.** When we score a house with bumped features, the NN predicts for a configuration that may or may not exist in training data. For typical homes this is reasonable interpolation; for edge cases (already at Overall Qual 10, or near the size limits) it's extrapolation, and predictions get unreliable.
- **No confidence intervals.** Both uplift and cost are point estimates. The NN has about \$30K test RMSE; that uncertainty propagates into the profit estimates and isn't quantified.
- **Hyperparameter search was narrow.** We tuned `alpha` and `learning_rate_init` but didn't touch `solver`, `activation`, or `batch_size`. Lbfgs in particular often outperforms Adam on small datasets like this one.

## Conclusion

The pipeline answers the original business question end to end. The neural network turns 8 features into a roughly \$30K-RMSE price estimate (test R^2 about 0.89), well above the mean-predictor baseline. The linear program then takes those predictions and a renovation cost model and picks the optimal subset under a fixed budget.

The prediction step is robust: NN-B beat 3 alternative architectures and held up across 12 hyperparameter combos, all chosen on a held-out val set. The optimization step is mathematically correct (LP returns Optimal status, constraints honored), but it inherits whatever cost assumptions we feed it. Under the 10%-of-value cost proxy, cheap homes win because they have the best profit-per-dollar. Under a flat \$30K cost, mid-tier homes win. The honest takeaway: prediction tells us what each house is worth; optimization tells us which ones to actually buy under a budget. The cost model is the main thing a real engagement would need to tighten.
