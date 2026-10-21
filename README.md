# Predictive Pipeline -- ETAI
20260548 - João Fernandes
week 2 
The best model is the logistic regression because it has better prevision capabilities without overfitting
week 3
## Conclusion: raw vs. cleaned dataset

Across both models and both feature sets (with/without `race`), cleaning cost almost no accuracy (at most -0.019, within normal test-split noise) but consistently narrowed the fairness gap -- in one case (decision tree, with `race`) from the worst possible FPR gap (1.000) down to 0.268.

The reason: on the raw path, uncleaned category variants (`"felony"`/`"f"`, placeholder tokens) fragment both the model's features and the fairness audit's groups. Cleaning doesn't add predictive signal -- it removes noise that was inflating raw accuracy through overfitting to duplicate categories, while letting the fairness audit finally group the same people together.

**Bottom line:** for this dataset, cleaning is nearly free in accuracy and clearly worthwhile for fairness.

Conclusão (week 4): a validação cruzada confirma o modelo escolhido nas semanas 2-3 -- a logistic regression continua a ser a melhor opção, com o menor gap treino-validação (praticamente zero) e a accuracy mais estável entre folds. A comparação holdout vs. CV mostra também porque a avaliação de uma única divisão não chega: os valores de holdout (uma única partição) e a média de CV diferem até 0.02 no mesmo modelo, só por causa de qual 25% calhou na validação -- é essa instabilidade que a CV existe para resolver. A decision tree e o random forest continuam a sobreajustar (gap de +0.10 e +0.09), o que sustenta a decisão de não os escolher sem antes limitar a sua complexidade.
## Overview

Predicts `two_year_recid` -- whether a defendant will be rearrested within
two years -- from ProPublica's COMPAS dataset: the data behind a real 2016
investigation into a risk-assessment algorithm actually used by US courts
to help inform bail and sentencing decisions. See `data/README.md` for the
full problem description and data dictionary.

It's interesting for reasons beyond plain accuracy: `race` is deliberately
excluded from the model's own inputs, kept aside only to check afterward
whether the model is equally wrong across groups, in the same direction
ProPublica's original investigation raised about COMPAS itself.

The pipeline started with some **deliberately weak spots**. Part of the
semester's work is finding them and making them better -- see the pipeline
progress table below, which tracks what changed and why as the weeks go
on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── cleaning.py         # clean_dataset -- category cleanup, domain rules, dedup, redundant columns
│   ├── preprocessing.py    # train/test split + one-hot encode
│   ├── model.py             # model construction + build_pipeline (preprocessing + model)
│   ├── tuning.py           # Optuna tuning + nested cross-validation
│   ├── evaluate.py         # accuracy metrics + fairness check + before/after comparison
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | From EDA to a validated cleaning recipe | Full EDA in `01_eda_introduction.ipynb`: missingness (including hidden placeholder tokens `"-"`/`"?"`), category aliases beyond simple case/whitespace (`"felony"`→`"f"`, `"african american"`→`"african-american"`), domain-rule violations, an outlier gap in `priors_count` (38 → 250/500, clearly sentinel values), a `decile_score`/`score_text` mutual-consistency check, redundant/multicollinear columns, and a full pairwise association matrix (Pearson + Cramer's V + correlation ratio) used to pick the final feature set. `clean_dataset()`, in its own `src/cleaning.py`, applies every one of these decisions (see "Preprocessing decisions" below). `main.py` runs **both** the naive path (`preprocess()` only, on the raw data) and the cleaned path (`clean_dataset()` then `preprocess()`) through **both** models and prints/saves a before-vs-after comparison table (`evaluate.compare_before_after`). See "Preprocessing decisions" and "Best model" below for what that comparison found. |
| 4 | Leak-free evaluation | Preprocessing and model joined in one sklearn `Pipeline`, cross-validated as a whole (stratified 5-fold) on the development set; a locked test set that no run touches; out-of-fold classification report and fairness audit. |
| 5 | Hyperparameter tuning | `src/tuning.py`: Optuna (seeded TPE sampler) searches the hyperparameters of the whole Pipeline, with search spaces in `config.yaml` (`tuning:`). The score to report comes from **nested CV** (tuning repeated inside each outer fold), plus an optimism check (tuning score minus honest score). |
| 6 | Ensembles | New models: `oblique_random_forest` (random forest whose trees split on random linear combinations of features) and `adaboost` (boosted trees of tunable depth). Their search spaces are in `config.yaml`. |
| 7 | Metrics and boosting | New models `gradient_boosting`, `hist_gradient_boosting`, `naive_bayes`. Several metrics per fold (`cv.metrics`), `roc_auc` as the selection metric, out-of-fold probabilities instead of labels, `class_weight: balanced` where the model supports it, a floor (dummy) and an untuned score for comparison, and a scoreboard `results/runs.csv` with one row per run. |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads the raw data, runs it through both the raw
(naive) and cleaned (`clean_dataset()`) preprocessing paths, trains every
model listed under `config.yaml`'s `models` section on each path, and
prints, per (dataset, model) combination:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

...then a before-vs-after summary table across all four runs (see "Best
model" below).

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Preprocessing decisions

Every decision below comes from `01_eda_introduction.ipynb` and is applied, step by step, by `clean_dataset()` in `src/cleaning.py` -- the module's docstrings reference the same findings.

| Column(s) | What was wrong | What was done | Why |
|---|---|---|---|
| `c_charge_degree`, `race` | Category spelled differently in a way plain case/whitespace cleanup doesn't catch: `"felony"` vs `"f"`, `"misdemeanor"` vs `"m"`; `"african american"` (space) vs `"african-american"` (hyphen) | Explicit alias map, applied after the usual strip+lowercase | These are real word variants, not formatting noise -- `.lower().strip()` alone leaves them as separate categories |
| `race`, `sex`, `priors_count`, `prior_offenses` | `"-"` / `"?"` used as missing-value markers, hiding inside otherwise valid-looking text | Converted to real `NaN` | Left as strings, these get one-hot encoded as if they were real categories (see the "raw" path's inflated feature count below) |
| `decile_score` | 6 rows outside COMPAS's own 1-10 scale (0, 15, 23) | Set to `NaN` | Domain rule -- physically impossible for that scale |
| `age` | 7 rows outside a plausible adult range | Set to `NaN`, then imputed with the global median | Domain rule; missingness here isn't concentrated in any one subgroup |
| `juv_fel_count` | 5 negative counts | Set to `NaN`, then imputed with `0` | A count can't be negative; the column is heavily skewed toward 0 already, so 0 is both the mode and a safe default |
| `priors_count` | 2 sentinel values (`250`, `500`, each appearing exactly 3 times) sitting in a hard gap above the real, continuous distribution (0-38) | Values `>60` set to `NaN` | The cutoff comes from the observed gap in the data, not a generic outlier rule -- the classic IQR rule here (≈12.5) would have discarded plausible values in the real tail |
| `decile_score` vs `score_text` | 9 rows where both are individually in-range but contradict each other (e.g. `decile_score=1` with `score_text="high"`) | Both set to `NaN` | No way to tell which of the two is wrong; picking one to trust would be arbitrary |
| `priors_count` (remaining ~487 missing, after placeholders + outliers) | Missingness high enough (~6.9%) that dropping rows would waste data | Imputed with the median **within each (`age_cat`, `c_charge_degree`) group** (global median as a fallback) -- all 6 groups have 300+ rows, comfortably enough for a stable median | Grouping deliberately excludes `race`: grouping the imputation by the sensitive attribute would manufacture a statistical dependency between race and priors_count that isn't in the real data, distorting the fairness audit before it even starts |
| `sex`, `race`, `c_charge_degree` | Missingness low (1.5-3.2%) with no defensible way to impute | Rows dropped | Demographic attributes have no genuine "N/A" case, and inventing a value for `race` (the sensitive attribute) would distort the fairness audit directly |
| `id` / exact row duplicates | 72 fully duplicated rows (same `id` twice) | Dropped, keeping the first occurrence | Same person entered twice |
| `prior_offenses`, `age_in_months`, `juvenile_total` | Exactly redundant: `prior_offenses` == `priors_count`; `age_in_months` == `age * 12`; `juvenile_total` == sum of the 3 juvenile counters, confirmed row by row | Dropped | Multicollinearity -- no new information, just noise for the model |
| `decile_score`, `score_text`, `age_cat` | Not wrong, but not model inputs | Excluded from `config.yaml`'s `features` list (not dropped from the dataframe) | `decile_score`/`score_text` are COMPAS's own prediction (Cramer's V 0.93 with each other, and the strongest association of anything with the target) -- using them would be target leakage, not modeling. `age_cat` has Cramer's V 0.88 with `age`, which already carries that information without discretizing it away |

The naive path (`preprocess()` on its own, no `clean_dataset()` first) still runs as the "raw" side of the before/after comparison below, so its cost stays visible rather than being silently replaced.

## Best model

`python main.py` trains logistic regression and a decision tree on both
the raw and cleaned data (4 runs total). Latest comparison:

| Model | Dataset | Rows used | Features | Train acc | Test acc | FPR gap (our model, by race) |
|---|---|---|---|---|---|---|
| Logistic regression | raw | 6256 | 56 | 0.675 | 0.666 | 0.333 |
| Logistic regression | cleaned | 6736 | 7 | 0.679 | 0.648 | 0.267 |
| Decision tree | raw | 6256 | 56 | 0.829 | 0.629 | 1.000 |
| Decision tree | cleaned | 6736 | 7 | 0.786 | 0.627 | 0.500 |

*(FPR gap = the spread between the highest and lowest false-positive rate across race groups on the test set -- 0 would mean the model is equally likely to wrongly flag every group; the smallest race groups, e.g. Asian/Native American with n=2 in a given split, make this noisier than the bigger groups.)*

**Conclusions:**
- **Cleaning did not raise test accuracy -- it lowered it slightly for both models** (-0.018 for logistic regression, -0.002 for the decision tree). That's expected, not a bug: with only 7 chosen features, `priors_count`/`sex`/`c_charge_degree` still loaded as text with `"-"`/`"?"` placeholders on the raw path, so `pd.get_dummies` treated every distinct spelling (`"male"`, `"Male"`, `"MALE"`, `" Male"`, and every `priors_count` value as its own category) as a separate column -- 56 features instead of 7. Some of that extra signal is real category information the model can exploit, just fragmented and inflated, not a genuine accuracy gain.
- **Cleaning substantially narrowed the fairness gap for both models** -- logistic regression's FPR gap dropped from 0.333 to 0.267, and the decision tree's dropped from a worst-case 1.000 (some group had every negative case wrongly flagged) to 0.500. Collapsing alias categories (`"felony"`/`"f"`, `"african american"`/`"african-american"`) into one spelling means the fairness audit is actually grouping the same people together, instead of splintering them across near-duplicate categories.
- **Feature count dropped from 56 to 7** -- the cleaned path's one-hot encoding only has `sex` and `c_charge_degree` (2 categories each) to expand, since `priors_count` is now a proper numeric column instead of ~40 text categories.
- **COMPAS's own score reproduces the original ProPublica finding**, independent of our cleaning: on the cleaned data, COMPAS's false-positive rate is 0.40 for African-American defendants vs. 0.25 for Caucasian defendants -- a real gap in the *label everyone is trying to predict/compare against*, not an artifact of our preprocessing.
- **Current best model: logistic regression on cleaned data.** It has the smallest train/test gap of the four runs (0.031, vs. 0.159 for the decision tree on the same data) and the best fairness result on real, correctly-grouped categories, at only a small accuracy cost versus its raw-data score. The decision tree still overfits heavily even on cleaned data (`max_depth: 1000` is still a deliberately unbounded, unfixed weak spot -- see the pipeline progress table) -- fixing that is next week's job, not this week's.

## Hyperparameter tuning (week 5)

Every number is ROC AUC (the selection metric, see week 7) on the development set, with the same stratified 5-fold split for every model. "Tuned" is the **nested CV** score (30 Optuna trials inside each outer fold), so the tuned model is always scored on rows that took no part in the choice. "Tuning score" is the best trial of the final study on all development rows; "Optimism" is the tuning score inside the outer folds minus the honest outer score.

| Model | Default CV (mean ± std) | Tuned nested CV (mean ± std) | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.596 ± 0.013 | 0.713 ± 0.012 | 0.717 | +0.004 | `max_depth=7`, `min_samples_leaf=87`, `criterion=gini` |
| Logistic regression | 0.719 ± 0.014 | 0.718 ± 0.014 | 0.719 | +0.001 | `C=0.0034` |
| Random forest | 0.664 ± 0.015 | 0.725 ± 0.014 | 0.726 | -0.000 | `max_depth=7`, `min_samples_leaf=56`, `max_features=0.5` |

Tuning matters most for the models that overfit by default: the decision tree goes from 0.596 to 0.713 just by limiting its depth and leaf size, and the random forest from 0.664 to 0.725. Logistic regression gains nothing (0.719 to 0.718): its default was already close to the best, and the search only chose a much stronger regularisation (`C=0.0034`) with the same score. The optimism is tiny for every model (at most +0.004), so with 30 trials and 5 inner folds the best trial's score is a fair estimate here; the nested score is still the one to report. Note that once the trees are regularised the gap between very different models almost disappears (all within about 0.01), which suggests this feature set limits the score more than the choice of model does.

## Ensembles (week 6)

| Model | Default CV (mean ± std) | Tuned nested CV (mean ± std) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.719 ± 0.014 | 0.718 ± 0.014 | 0.719 | `C=0.0034` |
| Decision tree | 0.596 ± 0.013 | 0.713 ± 0.012 | 0.717 | `max_depth=7`, `min_samples_leaf=87`, `gini` |
| Random forest | 0.664 ± 0.015 | 0.725 ± 0.014 | 0.726 | `max_depth=7`, `min_samples_leaf=56`, `max_features=0.5` |
| Oblique random forest | 0.669 ± 0.012 | 0.723 ± 0.014 | 0.726 | tree `max_depth=7`, `min_samples_leaf=55`, `n_components=4` |
| AdaBoost | 0.720 ± 0.012 | 0.727 ± 0.013 | 0.728 | tree `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

Averaging many trees (bagging) fixes most of the single tree's variance, and the forests end up about 0.01 above it, but tuned they are still only on par with logistic regression, so on this small, mostly low-signal feature set the extra flexibility buys little. The oblique forest, whose splits are random linear combinations of the features, does not beat the axis-aligned forest (0.723 vs 0.725, well inside one standard deviation): with only 7 mostly weak features there is little for oblique splits to exploit. AdaBoost is the best of this group, and it chose depth-4 trees with a small learning rate, i.e. a slow, regularised boosting rather than stumps. Its untuned score (0.720, stumps) was already good, which fits boosting needing little tuning to work.

## Metrics and scoreboard (week 7)

`python main.py` now appends one row per run to `results/runs.csv` (floor, untuned and honest score, fold scores, chosen hyperparameters), and reports accuracy, balanced accuracy, precision, recall, F1, F2, ROC AUC and average precision for every run (see the `results/run_<id>.txt` files). Scoreboard on ROC AUC, higher is better; "Tuned" is nested CV with 30 trials, "Untuned" uses the hyperparameters written in `config.yaml`:

| Model | Floor (dummy) | Untuned (CV) | Tuned (nested CV, mean ± std) |
|---|---|---|---|
| Dummy | 0.500 | 0.500 | - |
| Logistic regression | 0.500 | 0.719 | 0.718 ± 0.014 |
| Naive Bayes | 0.500 | 0.684 | 0.715 ± 0.011 |
| Decision tree | 0.500 | 0.596 | 0.713 ± 0.012 |
| Random forest | 0.500 | 0.664 | 0.725 ± 0.014 |
| Oblique random forest | 0.500 | 0.669 | 0.723 ± 0.014 |
| AdaBoost | 0.500 | 0.720 | 0.727 ± 0.013 |
| Gradient boosting | 0.500 | 0.725 | 0.728 ± 0.014 |

**Why ROC AUC is the selection metric.** Accuracy depends on one decision threshold (0.5) and on the class balance (45% reoffend), and it hides how the model ranks people. ROC AUC judges the ranking of the predicted probabilities, is threshold-free, and the threshold can then be chosen separately for the cost of each kind of error (for example F2 or recall if missing a reoffender matters more than a false alarm). It also lets the dummy floor be exactly 0.5, so every model's gain over doing nothing is obvious.

**Reading the board.** The boosting models are the top group (0.727-0.728), followed closely by the forests and logistic regression (0.718-0.725); the differences are smaller than one fold standard deviation (about 0.013), so none of them is clearly better than the others on this data. Because of that, simplicity should break the tie: logistic regression is within noise of the best model and is the easiest to explain and audit, which matters for a fairness-sensitive problem like COMPAS. The test set stays locked until a final model is chosen.

## Dataset

See `data/README.md`.
