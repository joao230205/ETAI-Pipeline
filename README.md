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
├── main.py                # entry point: clean, evaluate (CV / nested CV), refit, save the model
├── predict.py             # predicts a new CSV with the saved model (week 10)
├── config.yaml            # all tunable settings for COMPAS live here
├── config_adult.yaml      # the same pipeline on the Adult income data (week 9)
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # clean_dataset (row-preserving), drop_duplicate_rows (training only),
│   │                       # feature/target split, locked test set, build_preprocessor (imputers, encoders, scalers)
│   ├── model.py            # the models, VoteSelector (feature selection), build_pipeline (prep + select + model)
│   ├── tuning.py           # Optuna tuning + nested cross-validation
│   ├── evaluate.py         # scorers, CV of the whole pipeline, reports, fairness audit, corrected paired test
│   └── results.py          # saves each run's report and the scoreboard (runs.csv)
├── results/, results_adult/  # created automatically -- reports + scoreboard per config (not tracked in git)
├── models/                 # created automatically -- the saved pipelines (not tracked in git)
├── predictions_adult.csv   # predictions for data/adult_new.csv (week 9)
└── data/
    ├── compas_two_year_recidivism.csv, compas_new_people.csv
    ├── adult_train.csv, adult_new.csv, ADULT_README.md
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
| 8 | XGBoost and feature selection | New model `xgboost` (with its search space, including the new brakes `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`). New learned step `VoteSelector` between preprocessing and model (a random forest, gradient boosting and an L1 logistic regression vote; a feature needs `min_votes` votes), switched on in `config.yaml` (`feature_selection`), re-fitted inside every fold, and with tuning on `select__min_votes` is tuned together with the model (nested CV). The pipeline caches fitted steps in a temporary folder per run, so tuning doesn't refit the selector in every trial. The features kept are printed and recorded. Also fixed: the week-7 `main.py` had been saved in `src/` by mistake. See "Feature selection". |
| 9 | Beyond COMPAS (option A: Adult income) | `config_adult.yaml` runs the same pipeline on a new dataset (`python main.py --config config_adult.yaml`), with its own output folders. Fixed the COMPAS-only assumptions it exposed: placeholder tokens now become missing in every text column (not only the listed ones), and the COMPAS score in the fairness report is a config setting (`data.baseline_score`) instead of a hardcoded `score_text`. COMPAS gives exactly the same cleaning as before. Predictions for `adult_new.csv` in `predictions_adult.csv`. See "Beyond COMPAS". |
| 10 | Missing values and deployment | The numeric imputer is a setting (`imputation.numeric_strategy`: `median`, `mean`, `knn`, `iterative` (MICE), `random_forest` (missForest)); numeric columns are now scaled first, then imputed. `main.py` saves the final fitted pipeline (`models/model.joblib`), and the new `predict.py` predicts a new CSV from the terminal with no training, one row out per row in. See "Missing values" and "Predicting new data". |

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

```bash
python main.py                                                    # COMPAS (config.yaml)
python main.py --config config_adult.yaml                         # Adult income (week 9)
python predict.py data/compas_new_people.csv predictions.csv      # predict new people (week 10)
```

`main.py` loads the config, cleans the data (`clean_dataset()`, row-preserving), drops duplicate
rows (training data only), sets aside the **locked test set** (never touched), and evaluates the
model in `model.type` on the development set:
- the floor (a dummy model) and the model with the hyperparameters in the config, by stratified 5-fold CV of the whole Pipeline (preprocessing + feature selection + model);
- with `tuning.enabled`, Optuna tuning scored honestly by **nested CV** (the number to report);
- every metric in `cv.metrics`, a classification report and the fairness audit on the out-of-fold predictions;
- the final pipeline refit on every development row, the **features it kept**, and the pipeline saved to `output.model_path` for `predict.py`.

Each run adds one row to the scoreboard (`results/runs.csv`) and writes its full report to `results/run_<id>.txt`, so it doesn't just scroll past in your terminal.
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

## Feature selection (week 8)

XGBoost, tuned (30 Optuna trials, nested CV, ROC AUC), the same folds, with and without the `VoteSelector`. With selection on, `select__min_votes` is tuned together with the model.

| Run | Features used | Untuned (CV) | Tuned (nested CV, mean ± std) |
|---|---|---|---|
| XGBoost, no selection | 9 (all) | 0.723 | 0.728 ± 0.015 |
| XGBoost + `VoteSelector` | 2: `age`, `priors_count` | 0.721 | 0.725 ± 0.013 |

Corrected paired t-test (Nadeau & Bengio), selection minus no selection: **-0.002, t = -1.17, p = 0.31**.

The two-feature model is as good as the nine-feature one: the difference is about a fifth of a fold standard deviation and the corrected test is nowhere near significant, so dropping seven of nine features costs nothing measurable. That fits everything since week 5: `age` and `priors_count` carry almost all the signal, and the juvenile counts, sex, charge degree and the two missingness flags add little on top. Nested CV chose `min_votes = 1` for the final model and it still kept only those two -- the other columns didn't get a single vote. I would keep selection **on**: the same score with a model that only looks at age and criminal history is much easier to explain and audit, and it also means the model no longer uses `sex` at all, which matters for a fairness-sensitive problem like this one. The cost is a slower run (the selector fits three models per fold), which the cache keeps reasonable (about 2.5 minutes per tuned run here).

## Beyond COMPAS (week 9, option A: Adult income)

**Run it:** `python main.py --config config_adult.yaml` (results in `results_adult/`, model in `models/adult_model.joblib`), then `python predict.py data/adult_new.csv predictions_adult.csv --config config_adult.yaml`.

Decisions in `config_adult.yaml`, all from the data card: `sex` is the audited attribute; `race` is kept out of the features as well (the same "fairness through unawareness" choice as for COMPAS); `fnlwgt` (a sampling weight) and `id` are never features; `education` is redundant with `education_num`; `workclass` and `occupation` get a missingness flag (the same people are missing in both and earn > 50K half as often); Adult's own validity rules (age 17-90, so the 74 seventeen-year-olds survive). The 25 undocumented `v` columns all go in, and feature selection decides.

**What broke, why, and how it was fixed:**

| What broke | Why | Fix |
|---|---|---|
| `"?"` survived cleaning as a real category in `workclass`, `occupation` and `native_country` (silently -- no error) | `_canonicalize_categories()` only turns placeholders into missing values in the columns listed in `canonical_categories`; on COMPAS every text column happened to be listed | `clean_dataset()` now turns `placeholder_tokens` into NaN in **every** text column. After cleaning: 379 / 381 / 86 missing, as expected |
| ...and so the `workclass_was_missing` / `occupation_was_missing` flags were all 0 | the flag looks for NaN, and there was none | fixed by the same change |
| `KeyError: 'score_text'` in the fairness report | the comparison with COMPAS's own score was hardcoded (`df["score_text"] != "Low"`), and `split_features_target()` also named `score_text` | the external score is a setting, `data.baseline_score` (COMPAS: `{column: score_text, negative_value: Low}`; Adult: `null`), and the report shows it only when it exists |
| the fairness report said "did NOT reoffend" | COMPAS wording in a general function | neutral wording: "share of truly negative people who were predicted positive" |
| `canonical_categories: {}` / missing | assumed every dataset needs a category map | treated as empty |

Check: COMPAS's cleaned data is **identical** to before (`DataFrame.equals` on the old and new `clean_dataset()` output), so the COMPAS runs are unchanged. `main.py` already took `--config` (week 7).

**Results** (development set, 5-fold CV; tuned = nested CV, 30 trials; target scores: ROC AUC ≥ 0.90, accuracy ≥ 0.855, F1 ≥ 0.66):

| Run | Features kept | ROC AUC | Accuracy | F1 (class 1) |
|---|---|---|---|---|
| Floor (dummy) | - | 0.500 | 0.76 | 0.00 |
| Logistic regression + selection, untuned | 8 | 0.896 ± 0.015 | 0.847 | 0.640 |
| XGBoost, no selection, tuned | 37 (all) | 0.906 ± 0.015 | 0.859 | 0.661 |
| XGBoost + selection, tuned (`config_adult.yaml`) | 8 | 0.908 ± 0.015 | **0.856** | **0.668** |
| **XGBoost + selection, untuned** (n_estimators 300, learning rate 0.05, depth 4) | 8 | **0.913 ± 0.016** | **0.864** | **0.686** |

Both XGBoost runs with selection reach all three targets in the same run; the untuned one is the best on every metric. Corrected paired tests on ROC AUC: tuned with vs without selection +0.002 (p = 0.26); tuned vs untuned (both with selection) -0.005 (p = 0.07) -- neither difference is significant.

**Features kept (8):** `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `marital_status`, `occupation`, `relationship`. **None of the 25 `v` columns**, and neither `workclass`, `native_country` nor the two missingness flags. `predictions_adult.csv` holds `id`, `probability` and `prediction` for all 2,000 rows of `adult_new.csv` (19% predicted > 50K), from the saved tuned model.

The hardest assumption to find was the placeholder one, because nothing failed: `"?"` simply became a category and the missingness flags stayed at 0, and the model still scored well, so only checking `value_counts()` after cleaning showed it (the `KeyError` on `score_text`, by contrast, announced itself). What got the targets was a flexible model plus selection: logistic regression misses accuracy and F1, and XGBoost on all 37 columns only scrapes past F1 (0.661) because the 25 noise-like `v` columns cost it a little; with selection it keeps the eight real census features and clears all three targets. Tuning did not help here (0.908 vs 0.913, within noise): the hand-picked hyperparameters were already good, and nested CV is honest about the tuner not finding better. On fairness: `sex` is not a feature, but `relationship` (`Husband` / `Wife`) and `marital_status` carry it almost entirely, and the model wrongly predicts > 50K for 9% of men who don't earn it vs 2% of women -- leaving the sensitive attribute out does not make a model blind to it.

## Missing values (week 10)

The numeric imputer is now a setting (`preprocessing.imputation.numeric_strategy`). None of the options uses the target (filling a gap "by class" would leak the label in CV and can't run on a new person, who has no label), so all of them run unchanged in `predict.py`. XGBoost with the configured hyperparameters (tuning off), same folds, ROC AUC:

| Imputer | With selection (2 features kept) | p vs median | Without selection (9 features) | p vs median |
|---|---|---|---|---|
| `median` (default) | 0.721 ± 0.013 | - | 0.723 ± 0.017 | - |
| `knn` (5 neighbours) | 0.721 ± 0.010 | 0.94 | 0.721 ± 0.013 | 0.49 |
| `iterative` (MICE) | 0.721 ± 0.013 | 0.92 | 0.724 ± 0.017 | 0.82 |
| `random_forest` (missForest) | 0.721 ± 0.013 | 0.95 | 0.723 ± 0.015 | 0.99 |

(p = corrected paired t-test against `median` on the same folds.) Run times: median / knn / iterative under 15 s, random_forest about 90 s.

No imputer is better than the median: every difference is at most 0.003 with p ≥ 0.49. That's what the class notebook predicted: a model-based imputer is only as good as the columns it sees, and our five numeric columns say very little about each other (the juvenile counts don't tell you someone's age, and nothing predicts `priors_count` well). I keep **`median`**: same score, the fastest, nothing to tune, and the easiest to explain; the `_was_missing` flags already carry the information in the gaps that matter (`priors_count`, `c_charge_degree`). Note that the numeric columns are now scaled before they are imputed (needed for KNN); with the median this gives the same CV score as before (0.721 untuned, 0.725 tuned).

## Predicting new data (week 10)

```bash
python main.py                                                    # train, evaluate, save models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv      # predict, no training
python predict.py data/adult_new.csv predictions_adult.csv --config config_adult.yaml
```

`main.py` saves the final fitted pipeline (cleaning rules aside: imputer statistics, encoder, scaler, selected features and model) to `output.model_path`. `predict.py` cleans the new file with the same `clean_dataset()` rules but **without** `drop_duplicate_rows()`, so every row gets an answer, and writes a CSV with one row per input row: `id`, `probability` (of the positive class: reoffending within two years / earning > 50K) and `prediction` (1 if the probability is ≥ 0.5). On `compas_new_people.csv`: 20 rows in, 20 rows out -- the `"?"` in `priors_count` and the missing age are filled by the saved imputer, the never-seen charge degree `"X"` doesn't crash the encoder, and the same person entered twice gets the same prediction twice. It stops with a clear message if the input lacks a column the model needs. A `.joblib` file should only be loaded with the same library versions it was saved with, and only if you trust it (loading can run code).

## Dataset

See `data/README.md`.
