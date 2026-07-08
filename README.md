# Predicting Zombie Firms One Year Ahead

**A machine-learning pipeline that forecasts corporate zombification from financial statements — China & Japan listed firms, 2016–2024**

This project builds a binary classifier that predicts whether a firm will become a *zombie* in the **following** year, using ~50,000 firm-year observations from Chinese and Japanese listed companies. It grows out of an MPhil dissertation at the University of Cambridge on government support and systemic risk, and reuses the zombie-identification theory from that research (the FN-CHK method) as **predictive features**.

> 🇯🇵 日本語: [README.ja.md](README.ja.md) ／ 🇨🇳 中文: [README.zh.md](README.zh.md)

---

## Why this project is more than "fitting a model"

| Design choice | What it demonstrates |
|---------------|----------------------|
| **Leakage-proof target** | The zombie label is computed *from the current year's* financials, so predicting the current year leaks the answer. I shift the label forward one year (`t` features → `t+1` label), forcing the model to genuinely forecast. |
| **Domain knowledge as features** | The interest-rate-gap theory (CHK method) from my dissertation becomes a predictive feature — bridging econometrics and ML. |
| **Class imbalance** | Positives (zombies) are ~8%. Handled with `class_weight` / `scale_pos_weight`; evaluated primarily by **PR-AUC**, not accuracy. |
| **Time-based split** | Train on the past (2016–2020), test on the future (2021–2022) — no random split, mirroring real deployment. |
| **Explainability** | SHAP reveals which financial signals drive predictions, and whether they agree with economic theory. |
| **Error analysis** | I dissect false positives / negatives to understand *where* and *why* the model fails. |

---

## Key Results

| Model | AUC | PR-AUC |
|-------|-----|--------|
| Logistic Regression (baseline) | 0.927 | 0.591 |
| **Random Forest (best)** | **0.954** | **0.672** |
| XGBoost (tuned) | 0.952 | 0.663 |
| Random baseline | 0.500 | 0.095 |

PR-AUC of 0.67 is roughly **7× the random baseline** (0.095) — practical accuracy for forecasting zombification one year out.

### Findings that go beyond the metric

1. **Features beat model complexity.** Careful, theory-driven feature engineering lets even logistic regression reach PR-AUC 0.59; the most complex model adds little. In this problem, domain-informed features matter more than model choice.
2. **Theory validated by ML.** SHAP puts leverage (debt ratio) and profitability (ROA) at the top, with interest-based features — including the dissertation's **interest-gap (CHK)** — in the top six. Economics and ML agree.
3. **Zombification is cross-national.** Country (`is_china`) ranks near the bottom in SHAP importance: the financial signature of a zombie firm largely transcends borders.
4. **The model's blind spot is informative.** Missed zombies (false negatives) cluster among firms whose surface financials are improving but that still rely on hidden support — underscoring why "essence" features like the interest gap are needed.

---

## Repository Structure

```
.
├── notebooks/
│   └── zombie_prediction.ipynb     # End-to-end analysis (data → model → SHAP → error analysis)
├── data/
│   └── ml_dataset_features.csv     # Anonymized, ratio-based features (safe to share)
├── figures/
│   ├── shap_importance_bar.png     # SHAP feature-importance ranking
│   └── shap_beeswarm.png           # SHAP direction plot (validates economic logic)
├── requirements.txt
└── .gitignore
```

---

## Method Overview

**Target.** For each firm-year `t`, the label is whether the firm is a zombie (FN-CHK) in year `t+1`. Zombie identification combines an interest-rate-gap condition with profitability and leverage criteria, using country-specific policy rates (PBoC for China, BoJ average contract rates for Japan).

**18 features across five families.** Debt structure (leverage, term structure), interest & coverage (implied rate, **interest gap**, coverage), profitability (ROA, loss flag), dynamic trends (growth rates, consecutive-loss years, ROA volatility), and country. All ratios, so firms of any size compare fairly; extremes handled via inf→NaN and 1%/99% Winsorization.

**Models.** Logistic regression (median-imputed + standardized) as a baseline; random forest and XGBoost (which handles missing values natively) as the main models. Imbalance addressed by weighting; hyperparameters tuned via grid search with cross-validation **inside the training set only**.

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebooks/zombie_prediction.ipynb
```

- **Parts 1–3** (raw-data processing) require the licensed source files and won't run without them.
- **Parts 4–7** (modeling, SHAP, error analysis, tuning) run directly from the included `data/ml_dataset_features.csv`.

---

## Data & Privacy

Firm-level raw data (CSMAR for China, Osiris for Japan) is proprietary and **excluded**. Only the anonymized feature set is shared: firm identifiers are opaque codes (e.g. `CN_000001`), and all values are computed ratios — no company names, no raw financial amounts.

---

## Author

**Yan Hongsheng (严宏胜)** — MPhil, University of Cambridge
GitHub: [@hshyan0623](https://github.com/hshyan0623)

Companion repository (the econometric study this builds on): [systemic-risk-zombie-firms](https://github.com/hshyan0623/systemic-risk-zombie-firms)

---

*Shared for academic and portfolio purposes. Licensed datasets are excluded per their terms of use.*
