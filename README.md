# COMPAS Recidivism — Disparate Impact Audit

**DNSC 6330: Responsible Machine Learning**  
Individual Homework 3

> **Generative AI Disclosure:** Generative AI tools were used as a learning aid during the development of this work — specifically for brainstorming code structure, translating R syntax to Python, and reviewing outputs for accuracy. All AI-generated content was critically reviewed, validated, and integrated as the author's own intellectual product. This disclosure is made in accordance with GW's Generative AI Use Policy.

## Purpose

This repository contains the Individual Homework 3 submission for DNSC 6330, building directly on the Lecture 03 live-coding exercise and extending the COMPAS pipeline from Homeworks 1 & 2. The goal is to conduct a rigorous disparate impact audit of a Gradient-Boosted Tree (GBT) recidivism classifier, quantifying algorithmic bias across racial and sex subgroups using legally grounded fairness metrics.

The workflow covers:

1. **Data loading and preprocessing** — Same ProPublica COMPAS dataset and ProPublica filtering rules as HW1 & HW2.
2. **Model training** — Logistic Regression (GLM) and Gradient-Boosted Tree (GBT) fitted following the Lecture 02 code slides, providing continuity across the homework series.
3. **AIR, ME, and SMD** — Adverse Impact Ratio, Marginal Effect, and Standardized Mean Difference computed for race and sex separately using the `solas-ai-disparity` Python library, with confirmation that both calls use an identical methodology.
4. **Intersectional analysis** — Combined (race × sex) subgroups evaluated against `Caucasian_Male` as the reference, with worst-group AIR identified and interpreted.
5. **FPR and FNR disparities** — Per-race False Positive Rate and False Negative Rate computed from confusion matrices; each compared to the Caucasian baseline using a two-proportion z-test for statistical significance.
6. **Publication-quality figure** — Grouped bar chart of FPR and FNR by race with Caucasian dashed reference baselines, saved at dpi=150.
7. **300-word compliance memo** — Addressed to a hypothetical federal regulator, summarizing metrics used, key findings, and limitations.

## Python Libraries Used


| Library        | Purpose                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `pandas`       | Data manipulation and tabular analysis                                       |
| `numpy`        | Numerical computations and array operations                                  |
| `matplotlib`   | Publication-quality visualization                                            |
| `seaborn`      | Statistical plot styling                                                     |
| `scikit-learn` | Preprocessing pipelines, Logistic Regression, GBT, and confusion matrices    |
| `statsmodels`  | Two-proportion z-test (`proportions_ztest`) for FPR/FNR significance testing |
| `solas-ai`     | AIR, ME, and SMD disparity metrics (legally grounded library)                |


## Instructions for Reproducing the Results

### Prerequisites

- Python 3.9 or later
- `pip` package manager
- Internet connection (the notebook downloads the COMPAS dataset automatically)

### Steps

1. **Clone this repository:**
  ```bash
   git clone <repository-url>
   cd <repository-name>/HW3
  ```
2. **Install dependencies:**
  ```bash
   pip install -r requirements.txt
  ```
3. **Run the Jupyter Notebook:**
  ```bash
   jupyter notebook DNSC-6330-Responsible-ML-homework-03.ipynb
  ```
   Alternatively, open the notebook in JupyterLab, VS Code, or Google Colab and run all cells sequentially (Runtime → Run all).
4. **Data source:** The notebook downloads the dataset automatically from the ProPublica GitHub repository — no manual download is required.

> **Note on runtime:** Training the GBT with 200 estimators takes approximately 2–3 minutes on a standard laptop CPU. SHAP computation is not required for this homework. The full audit pipeline completes in roughly 3–5 minutes.

## Repository Structure

```
HW3/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
└── DNSC-6330-Responsible-ML-homework-03.ipynb  # Main analysis notebook
```

## Data Source

Broward County COMPAS scores and two-year recidivism outcomes from the [ProPublica Machine Bias investigation](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) (2016).

- Dataset: `[compas-scores-two-years.csv](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)`
- Original R analysis: [ProPublica/compas-analysis](https://github.com/propublica/compas-analysis)

## Key Findings Summary

- African-American defendants exhibit the lowest favorable-outcome rate (predicted non-recidivism), with an AIR below the EEOC 80% threshold relative to Caucasian defendants.
- The FPR disparity for African-American defendants is statistically significant (p < 0.05), replicating the ProPublica finding that Black defendants are disproportionately labelled high-risk.
- Intersectional analysis reveals that **African-American Male** defendants face the worst compound AIR (AIR = 0.576) relative to Caucasian Males — a harm that single-axis race-only analysis would obscure.
- The model cannot simultaneously satisfy calibration and FPR/FNR parity when base recidivism rates differ across groups (Chouldechova 2017 impossibility theorem). Any deployment decision requires an explicit, documented choice of which fairness criterion to prioritise.

## References

- Gill, J., Hall, P., Montgomery, K., & Schmidt, N. (2020). A Responsible Machine Learning Workflow. *Information*, 11(3), 137.
- Chouldechova, A. (2017). Fair Prediction with Disparate Impact. *Big Data*, 5(2), 153–163.
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. fairmlbook.org.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS 2016.

