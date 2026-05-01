# Can Socioeconomic and Demographic Factors Predict High School Academic Performance?

**DSCI 303 Final Project — Rice University**

**Authors:** Isabella Barone (`imb5@rice.edu`), Allie Meachum (`am375@rice.edu`)

---

## Overview

This project investigates whether socioeconomic and demographic factors alone are strong enough to reliably predict high school academic performance. Using a Kaggle dataset of 1,000 American high school students, we evaluate six models (Linear Regression, Logistic Regression, Random Forest, Neural Network, K-Means, and a zero-shot LLM via Gemini 2.5 Flash) and compare them using ANOVA and post-hoc Tukey HSD on 5-fold cross-validated F1 scores.

**Main finding:** Demographic features alone explain only ~17% of test score variance ($R^2 = 0.17$). All supervised models converge at an F1 ceiling of approximately 0.47. The prediction ceiling is determined by the weak information content of demographic features rather than by model choice.

---

## Repository Contents

```
.
├── Final_Project_High_School_Test_Results.ipynb   Main analysis notebook
└── README.md                                       This file
```

---

## Dataset

We use the publicly available [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset from Kaggle. It contains 1,000 students with the following features:

- `gender`
- `race/ethnicity` (groups A through E)
- `parental level of education`
- `lunch` (standard or free/reduced — used as an income proxy)
- `test preparation course` (completed or none)
- `math score`, `reading score`, `writing score` (0–100)

The notebook downloads the dataset directly; no manual file placement is required.

---

## How to Run

### Option 1: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Final_Project_High_School_Test_Results.ipynb`
3. Run all cells (Runtime → Run all)

The Gemini API cells require Google Cloud authentication. When prompted, sign in with a Google account that has access to Vertex AI / the `dsci303-test` project.

### Option 2: Local Jupyter

```bash
# Clone this repo
git clone https://github.com/imb5-rice/dsci303-students-performance.git
cd dsci303-students-performance

# Install dependencies
pip install -U pandas numpy scikit-learn matplotlib seaborn imbalanced-learn statsmodels google-genai jupyter

# Launch Jupyter
jupyter notebook Final_Project_High_School_Test_Results.ipynb
```

---

## Methods Summary

### Preprocessing
- Created `avg_score` (mean of math, reading, writing) and binary `high_performer` label (top 25%, threshold = 77.67)
- Ordinally encoded parental education (0 = some HS, 5 = master's)
- One-hot encoded other categoricals; standardized all features

### Feature Selection
- Mutual information (MI) against `avg_score`
- Selected features with MI > 0.01 for reduced-feature Random Forest

### Class Imbalance
- SMOTE oversampling on training set (254 high performers vs. 746 non-high performers)

### Models

| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline; provides $R^2$ ceiling |
| Logistic Regression | Linear classification benchmark |
| Random Forest | Captures compounding SES interactions |
| Neural Network (MLP) | Tests whether non-linear learner finds extra signal |
| Gemini 2.5 Flash (LLM) | Zero-shot test using world knowledge |
| K-Means (k=6) | Unsupervised demographic-tier check |

### Evaluation
- 5-fold stratified cross-validation
- ANOVA + post-hoc Tukey HSD ($\alpha = 0.05$) for model comparison
- Same statistical tests applied to validate cluster differences

---

## Key Results

- **Linear Regression:** $R^2 \approx 0.17$ — demographics explain ~17% of score variance
- **Top three predictors** (consistent across MI, RF importance, and LLM reasoning): **lunch status, parental education, test preparation**
- **Tukey HSD:** Logistic Regression, Neural Network, and Random Forest (Top-5 MI) are statistically tied at F1 ≈ 0.42–0.47; all significantly beat the baseline
- **Clustering:** K-Means with k=6 produces clusters with significantly different average scores (top: 72.8 / 38.6% high performers; bottom: 63.0 / 15.3% high performers)

---

## Limitations & Future Work

- Dataset contains only demographic features; including academic history (prior grades, attendance) would likely raise the prediction ceiling
- F1 ≈ 0.47 implies these models misclassify roughly half of high performers from disadvantaged backgrounds — they should not be the sole basis for resource allocation
- Future work should apply fairness-aware learning techniques before any deployment in real educational settings

---

## References

1. Sirin, S. R. (2005). Socioeconomic Status and Academic Achievement: A Meta-Analytic Review of Research. *Review of Educational Research*, 75(3), 417–453.
2. Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. *Proceedings of 5th FUBUTEC Conference*.
3. Gray, G., McGuinness, C., & Owende, P. (2014). An Application of Classification Models to Predict Learner Progression in Tertiary Education. *2014 IEEE International Advance Computing Conference*.

---

## License

This project is for educational purposes (DSCI 303 at Rice University). The Kaggle dataset has its own license — see the original dataset page for details.
