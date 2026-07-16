# NBA Prediction — Improvement Plan

> Status: **DRAFT FOR REVIEW** — no code will be written until this plan is approved.
> Produced by an agent team: 🏗️ Architect, 📊 Data Scientist. 🧪 Unit Tester and 🔎 Code Reviewer engage during execution.
> Direction agreed with the user: **refactor into a package · correctness first · plan doc before code.**

---

## 1. What the project is today

Three copy-pasted Jupyter notebooks that run the identical 9-stage pipeline to predict whether the **home team wins**:

- `01_data_cleaning.ipynb` — fetch 3 seasons (2021–24) via `nba_api`, engineer features, 80/20 time split, GridSearchCV-tuned XGBoost, isotonic calibration, save 3 `.pkl` files.
- `02_2425_season_test.ipynb` — reload, rebuild the same features for 2024-25, report accuracy/calibration/betting P&L.
- `03_inferemce_engine.ipynb` — same features for 2025-26, `predict_game()` for future matchups, Kelly betting calculator.

Reported ~67% accuracy / ~76% on high-confidence games. **These numbers are not trustworthy** — see §2.

The feature logic is correct in one important respect worth preserving: every rolling/cumulative feature is properly lagged with `.shift(1)` / `cumcount()`, so there is **no same-game target leakage**. The bugs below are about *skew* and *cross-season contamination*, not same-game peeking.

---

## 2. Bugs found (all verified against the code)

| # | Sev | Bug | Root cause |
|---|-----|-----|-----------|
| 1 | 🔴 | **Fatigue train/serve skew** | `01` uses `FATIGUE_3_IN_4 ≥ 2` / `4_IN_5 ≥ 3`; `02`/`03` use `≥ 3` / `≥ 4`. Model trained on one flag definition, scored on another. A window-anchoring error (`shift(1)` on the count) makes *both* versions slightly wrong. |
| 2 | 🔴 | **Cross-season leakage** | Every `groupby('TEAM_ID')` (CUM_WINS, GAMES_PLAYED, SEASON_WIN_PCT, DAYS_REST, all `ROLL_*`, fatigue windows, SOS) omits `SEASON`, so records/form bleed June→October. Also a *distribution* skew: training aggregates are 3-season-cumulative while `02`/`03` are single-season. |
| 3 | 🔴 | **Star feature uses wrong data** | `01` builds `star_map`/`active_pairs` from loop-leftover single-season (2023-24) vars instead of the `*_3seasons` concats → `DIFF_STAR_ACTIVE ≈ 0` for ~2/3 of training data. |
| 4 | 🟠 | **Calibration is broken** | Isotonic fit on **in-sample** train probs of the tuned model; a **separate untuned** model produces the test probs; the saved `.pkl` is a third object. `02`/`03` never even apply the calibrator at inference. |
| 5 | 🟠 | **CV leakage** | `GridSearchCV(cv=5)` = stratified K-fold on time-ordered data → future validates past → wrong hyperparameters, optimistic scores. |
| 6 | 🟡 | **Not reproducible** | No `random_state` anywhere; `subsample`/`colsample` are stochastic. Requirements unpinned. |
| 7 | 🟡 | **`predict_game()` extra skew** | Feeds `ROLL_SEASON_WIN_PCT` where training used raw `SEASON_WIN_PCT`; a *third* fatigue definition; returns uncalibrated probs; silently zero-fills missing features; `get_tomorrows_games()` uses `datetime.now()` (today, not tomorrow). |

**The correct fatigue definition** (to be used identically everywhere): count games in the trailing 4-/5-calendar-day window **including the current game, no shift**, then threshold `≥3` / `≥4`. This uses only the schedule (known pre-tip), so it is not leakage; it fixes the skew *and* the anchoring bug at once.

Non-bugs confirmed safe (do not touch): the `.shift(1)` lags, the 80/20 temporal split mechanics, and the SOS lookup (opponent win pct is itself past-only). Flagged for Phase 2, not correctness: `ADJ_NET_RAT = ROLL_NET_RAT * ROLL_SOS` is a rescaling, not a real opponent adjustment; the betting backtest uses a flat `100/180` payout (a toy).

---

## 3. Target architecture (`src/nba_pred/`)

The refactor's #1 job is **one feature pipeline** shared by train / test / inference, so skew (bugs #1, #7) *cannot structurally recur*.

```
src/nba_pred/
  config.py            # ANTI-SKEW KEYSTONE: seasons, ROLL_SPAN, fatigue thresholds,
                       # penalty coeffs, PARAM_GRID, RANDOM_STATE=42, betting params
  paths.py
  data/
    cache.py           # parquet cache (completed seasons immutable, current season TTL)
    ingest.py          # nba_api wrappers; every fetch stamps a SEASON column
  features/
    schema.py          # column constants + canonical FEATURES order
    team_features.py    # SINGLE SOURCE OF TRUTH for per-team-game engineering (groups by [SEASON, TEAM_ID])
    stars.py           # season-aware star map + DIFF_STAR_ACTIVE
    matchup.py         # home/away split, merge, DIFF_*, penalties -> (X, meta)
    state.py           # "as-of" latest-state extraction for FUTURE games
  model/
    split.py           # temporal train/calibration/test split
    train.py           # GridSearchCV + TimeSeriesSplit, scoring=neg_log_loss
    calibrate.py       # ONE isotonic/Platt calibrator on a held-out fold
    bundle.py          # ONE artifact: {model, calibrator, features, provenance}
  evaluation/
    metrics.py         # accuracy, Brier, log-loss, AUC, calibration table, high-conf + Wilson CI
    plots.py           # confusion-matrix / feature-importance figures
  betting/
    kelly.py  backtest.py  schedule.py
  pipelines/
    train.py  backtest.py  inference.py   # run_training/run_backtest/run_inference
scripts/  train.py backtest.py predict.py  # headless CLI (also console_scripts)
data/cache/…                               # raw pulls
artifacts/model_bundle.pkl                 # replaces the 3 mismatched pkls
pyproject.toml                             # pinned deps + dev extras + entrypoints
tests/                                     # pytest suite (§5)
```

**The parity guarantee:** one function `assemble_diff_row(home, away, config)` holds every fatigue threshold and the penalty formula. Batch training calls it over merged games; future inference calls it on synthesized rows from `state.py`. There is physically no second copy of a `>= N` comparison. The bundle stores `FEATURES`, so inference validates column order instead of silently zero-filling.

Notebooks become thin callers: `run_training()`, `run_backtest("2024-25")`, `run_inference("2025-26")` + `predict_upcoming()`, with plotting the only notebook-resident code.

---

## 4. Migration sequence (behavior-preserving, then fix)

Each step is one reviewable commit; §4's parity proofs are enforced by the 🔎 reviewer + 🧪 tests.

0. **Golden snapshot.** Run the 3 notebooks as-is against a frozen cache; save each season's `X` / predictions / metrics as the parity oracle.
1. **Scaffold** package + config + cached ingest. Prove ingest reproduces current raw frames (row counts, columns).
2. **Extract features bug-for-bug.** Reproduce present behavior *including* the skew/leakage, parametrized by threshold set. Proof: `assert_frame_equal(new_X, golden_X)` per season.
3. **Extract model/calibrate/bundle** reproducing the current save. Proof: identical `best_params_`, identical golden-test predictions.
   *→ At this point it is a pure refactor; outputs unchanged.*
4. **Fix #1 — unify fatigue** (include-current definition) everywhere. Assert only the two fatigue columns changed vs golden; retrain; record metric delta.
5. **Fix #2 — season-reset** all grouping to `[SEASON, TEAM_ID]`. Assert change is localized to each season's first-N games; retrain; log before/after.
6. **Fix #3 — star data** from the 3-season concat, keyed by `(SEASON, TEAM_ID)`. Assert `DIFF_STAR_ACTIVE` becomes non-trivial in 2021-22 & 2022-23.
7. **Fix #4/#5 — calibration + CV.** Three-way temporal split (train/cal/test); one calibrator on the held-out cal slice; `TimeSeriesSplit`, `scoring=neg_log_loss`; delete the stray untuned `.fit()`; inference always applies the calibrator. Assert disjoint train/cal/test index sets.
8. **Fix #7 — unify the future path.** Rebuild `predict_game` on `state.py` + `assemble_diff_row`; feed raw `SEASON_WIN_PCT`; apply calibrator; strict FEATURES validation. **Key test:** reconstruct a completed game's feature vector via both batch and inference paths and assert element-wise equality.
9. **Fix #6 — reproducibility:** `random_state=42` threaded through; pin deps; date-pinned split boundaries.

---

## 5. Test suite (🧪 Unit Tester, built alongside Phase 1)

- **Fatigue helper:** synthetic schedules (1-2-4, 1-2-3-5, 1-4-5) → flags equal hand-computed truth; `FATIGUE_3_IN_4.mean()` in a plausible ~0.15–0.30 band, not ~0.
- **No-leakage:** every `ROLL_*`/cumulative feature invariant to the *current* game's outcome (swap game i's result → feature[i] unchanged); forbid `KFold`/`shuffle=True` near time data.
- **Season reset:** first game per `(TEAM_ID, SEASON)` has `GAMES_PLAYED==0`, `CUM_WINS==0`, `ROLL_*` NaN; `GAMES_PLAYED` ≤ ~82/season.
- **Train/serve parity:** the "one vector, two paths" equality test (batch vs `predict_game`) — the single most valuable guard.
- **Calibration:** train/cal/test index sets disjoint; calibrated Brier/log-loss ≤ raw on test.
- **Betting math:** `suggest_bet_amount(0.55, …) == 0` (below gate); known p/odds → hand-computed half-Kelly, capped.

---

## 6. Honest re-baselining protocol (after fixes)

- **Splits:** Train = 2021-22 + 2022-23 · Calibration = first ~70% of 2023-24 · Test-1 = last ~30% of 2023-24 · **Test-2 (true OOS) = full 2024-25.**
- **Tuning:** GridSearchCV + TimeSeriesSplit on training seasons only, `scoring=neg_log_loss`.
- **Report the full panel** (not accuracy alone): Accuracy, **Brier, log-loss**, AUC, reliability curve + ECC, and high-confidence subset accuracy **with n and a Wilson CI** (the "76%" claim is meaningless without n).
- **Baselines to beat:** always-pick-home (~0.57–0.60), higher-season-win-pct, and an **Elo** model. If XGBoost doesn't beat home-court + Elo on log-loss, the features aren't earning their keep.
- **Expect the headline numbers to move** — treat 67%/76% as void until re-measured. README updated afterward.

---

## 7. Phase 2 — accuracy roadmap (only after correctness is locked)

1. **Elo baseline** (K≈20, home ≈ +100, MOV multiplier, 25% between-season regression) as the yardstick + as `ELO_DIFF`/`ELO_WIN_PROB` features.
2. **Real opponent-adjusted** net rating (ridge/APM-style) replacing the `ROLL_NET_RAT * ROLL_SOS` hack; add pace, home/away splits.
3. **Better availability** than a single top-scorer flag (top-3 by minutes / impact; pre-game injury report — *not* post-game logs, which leak).
4. **Model bake-off:** regularized logistic (strong calibrated baseline) vs XGBoost vs LightGBM vs small stack, compared on log-loss/Brier under the same temporal CV.
5. **Threshold optimization** on the calibration slice, evaluated on test (coverage vs accuracy sweep).
6. **Honest betting eval:** real historical closing odds + vig, fractional Kelly on *calibrated* edges, ROI vs **closing-line value** with bootstrap CIs. Never tune the threshold on the games you report ROI on.

**Leakage guards for Phase 2** (🔎 reviewer enforces): opponent-adjusted/SOS/Elo features must use strictly-prior games; availability must be as-of pre-game; calibrator never sees train or test; the "one vector, two paths" parity test gates every feature addition.

---

## 8. Scope decisions (agreed with user 2026-07-08)

1. **Betting:** fix the Kelly bugs (KELLY_CAP, feed *calibrated* probs) and keep the tool, but **defer** the honest ROI/CLV backtest until we can source historical odds data. `betting/backtest.py` stays a placeholder for now.
2. **Data cache:** **commit a frozen parquet cache** to the repo so training/tests are reproducible offline and CI never hits the flaky `nba_api`.
3. **Phase 2 depth:** go as far as **Elo baseline + honest re-baseline, then pause and reassess** with real numbers. Do NOT pre-build the full model bake-off this engagement.
```
