# Wine Quality — the regression baseline, and what it cost to beat it

**Five-person group project (CSCI 323 Modern AI).** Predicting wine quality scores from eleven
physicochemical measurements, on the two UCI wine datasets — 1,599 red and 4,898 white — with both a
classification half and a regression half.

**My part, as the team split it:**

| What | Scope |
|---|---|
| **Code** | **Linear Regression** — the model, the tuning, and its performance report. Random Forest and Gradient Boosting were coded by other members, as was the whole classification half. |
| **Report** | §2.2 data cleaning and preprocessing · §2.3 the three regression models · §3.2 regression model setup and fine-tuning |
| **Presentation** | Section 3 — introduction of the three regression models and their setup |

The comparison and analysis of the three regression models — section 4 of the presentation — was a
different member's section, and the classification half was two others'. `docs/` holds the
presentation with the team's contribution table, so the split can be read rather than taken on trust.

Writing the section that introduces all three models while coding only one of them turned out to be
the useful part of this project, because it meant reading the other two closely enough to explain
what they do differently — and then watching my own model land almost on top of them.

---

### 1. The baseline that was not beaten by much

Linear Regression is the model you include so the ensembles have something to beat. On this data they
barely did. From the notebook's comparison table, which is another member's section:

| Red wine | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|
| **Linear Regression** (mine) | 0.6560 | **0.6629** | 0.3703 | 0.3294 |
| Random Forest Regressor | 0.4468 | 0.6451 | 0.7079 | 0.3649 |
| Gradient Boosting Regressor | 0.5388 | 0.6441 | 0.5752 | 0.3670 |

| White wine | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|
| **Linear Regression** (mine) | 0.7443 | **0.7505** | 0.3003 | 0.2947 |
| Random Forest Regressor | **0.2564** | 0.7013 | **0.9170** | 0.3841 |
| Gradient Boosting Regressor | 0.5705 | 0.7109 | 0.5889 | 0.3672 |

Two things in those tables are worth reading together, and they are about the columns most model
comparisons drop.

**The ensembles win on test error by 0.02 on red and 0.05 on white.** That is the whole return on
replacing a closed-form fit with a tuned forest of hundreds of trees.

**My model is the only one whose two error columns are the same number.** Train 0.6560 against test
0.6629 on red, 0.7443 against 0.7505 on white — a gap of about 0.007 either way. The white-wine
forest has a gap of 0.445 and a training R² of 0.9170 against a test R² of 0.3841. It has
substantially memorised the training set, and what that bought over my baseline is 0.049 RMSE.

Linear Regression cannot overfit here because there is almost nothing in it to overfit with: eleven
coefficients and an intercept. That limitation is what makes it a measurement. The distance between
my train and test error is roughly the noise floor, and every model in the table is stuck near the
same test error, so the ceiling in this problem is the eleven features and not the choice of
algorithm.

---

### 2. Where the scaler sits

```python
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])

lr_grid = GridSearchCV(lr_pipeline, {"linreg__fit_intercept": [True, False]},
                       cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
```

The scaler is a step inside the pipeline rather than something applied to the data beforehand, and
that is the only detail in this block I would defend at length. `GridSearchCV` refits the whole
pipeline on each of the five training folds, so the mean and standard deviation used to standardise
come from four folds and are then applied to the fifth. Scaling the full training set first would
compute those two numbers from data the model is about to be scored on, and every cross-validation
number in this notebook would come out slightly optimistic.

§2.2 of the report is where I set out why the scaling is needed at all: Linear Regression fits
weights by minimising squared error, so a feature measured in hundreds pulls harder on the
optimisation than one measured in units, purely because of its range. The tree models do not need
it — a decision tree splits on a threshold within a single feature, and a threshold does not care
what the neighbouring feature's units are. That distinction is why the scaler appears in my pipeline
and not in the other two.

---

### 3. What the residuals say about the model I was asked to build

![Residual plot, Linear Regression, red wine](images/residuals_linear_regression_red.png)

The residuals spread wide around zero across the whole prediction range rather than tightening
anywhere, which is the signature of a model that is underfitting rather than one that is overfitting
or heteroscedastic. It is consistent with the train-test numbers above: the model is not failing on
unseen data, it is failing evenly everywhere, which is what a linear form does to a relationship that
is not linear.

![Linear Regression coefficients, red wine](images/coefficients_linear_regression_red.png)

For a linear model the coefficients are the interpretation — each one is the change in predicted
quality for a one-unit change in that feature with the others held constant, which is a statement the
tree models cannot make about themselves at all. Because the features were standardised first, the
coefficients are on a common scale and can be compared to each other directly.

---

### 4. The sections I wrote

`docs/regression_sections.docx` is my part of the report as submitted — three sections:

- **§2.2 Data cleaning and preprocessing.** No missing values in either dataset — the UCI copies are
  clean — but duplicates are not rare: red drops from 1,599 rows to 1,359 and white from 4,898 to
  3,961, so about a fifth of the white dataset is repeated rows. They are removed **before** the
  split, which is the part that matters: an identical row on both sides of the split is a free
  correct answer at test time. Then the StandardScaler argument above, with the formula and the
  reason the tree models are exempt.
- **§2.3 The three regression models.** What each one is and what it assumes, written for a reader
  who has not met them: the linear form and OLS, bagging and the averaging of independent trees, and
  boosting as sequential fitting to residuals with a learning rate controlling each tree's
  contribution.
- **§3.2 Setup and fine-tuning.** The search space for every model, the values the search chose on
  each dataset, and the cross-validated RMSE it reached.

Every figure in §3.2 is reproduced from the notebook's own cell outputs; I checked them against the
committed notebooks rather than transcribing from an earlier draft.

One result in that table is worth naming because it points the same way as section 1. The Random
Forest search space allowed `min_samples_leaf` of 1, 2 or 4 and `min_samples_split` of 2, 5 or 10.
On red wine the search chose 4 and 10 — constrained leaves. On white it chose 1 and 2 — no
constraint at all, which is the configuration that produced the 0.9170 training R². The same search
space, run on a dataset three times larger, walked to the opposite end of it.

---

### Repository Structure

```
wine-quality-regression/
├── WineQuality_RED.ipynb          # 68 cells. My Linear Regression block is cells 33-43.
├── WineQuality_White.ipynb        # 67 cells. Mine is cells 32-42.
├── wine+quality/                  # UCI data, red and white, plus the variable description
│                                  # notebooks read this path, so both stay where they were
├── docs/
│   ├── regression_sections.docx   # my report sections 2.2, 2.3 and 3.2
│   └── CSCI323_FT14_Presentation.pdf   # includes the team's contribution table
├── images/                        # two charts, extracted from the notebook's own outputs
└── README.md
```

### How to Run

```bash
git clone https://github.com/JinWanKim98/wine-quality-regression.git
cd wine-quality-regression
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook WineQuality_RED.ipynb
```

The notebooks sit in the repository root because that is where they expect `wine+quality/` to be.
Every cell output is saved in the committed files, so they read correctly without running anything.

### Provenance

Group project, five members, CSCI 323 at UOW (SIM Singapore), Semester 2 2026. The work was split by
task at the start: one member per regression model for the coding and tuning, one for the regression
comparison, and the classification half across the remaining members. I took Linear Regression.

The notebooks are the team's submitted files, unchanged. Teammates are named in
`docs/CSCI323_FT14_Presentation.pdf`, including on the contribution table, which is the point of a
contribution table. No student ID numbers appear anywhere in this repository. The demo presentation
was recorded by all five of us and is not included here.

### Limitations

- **I am not the author of the Random Forest or Gradient Boosting code, or of the comparison
  section.** Their numbers are quoted here from the committed notebooks because my model is only
  interesting next to theirs. The reasoning about what those numbers mean is mine; the models and the
  comparison write-up are not.
- **Tuning a linear regression is close to a formality.** The only hyperparameter searched was
  `fit_intercept`, and `True` won, which it almost always does. The grid search around my model is
  there for consistency with the other two, not because it had a decision to make.
- **The train-test gap is a weak measure of overfitting on its own.** It shows the white-wine forest
  memorising, but a small gap does not prove a model is well specified — mine has a small gap because
  it is too rigid to fit the training set closely in the first place.
- **Duplicate removal happened before the split**, which is right for preventing the same row landing
  on both sides, but it also means the test set is not a sample of the raw data as collected.
- **Both datasets are one region and one certification body.** Nothing here says how these models
  behave on wine scored by a different panel.
