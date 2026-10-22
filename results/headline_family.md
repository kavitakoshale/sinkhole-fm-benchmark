# Headline family (fixed in advance)

The four settings below form the confirmatory part of the paper (Table 2, Section 4.1). They were written into the
project's internal plan before the seed study and the tuned baselines were run. The section is reproduced below verbatim,
except that the path of the collected results was updated to `results/README.md`. The rest of the plan (schedule,
venues, internal notes) is not part of the release. The family was not registered externally. C1 is the plan's headline
claim, that a foundation model is at least as good as the tuned GBDT. C2 to C7 are its exploratory claims (label
efficiency, feature scarcity, label noise, calibration, transfer and cost), which the paper reports with intervals but
without family-wise correction.

---

## Pre-registered headline family

Fixed before the seed study and the tuned baselines were run (2026-10-21):

| Setting | Reference | Claim tested |
|---|---|---|
| Florida no_landcover × ring × spatial5 | xgboost_tuned (nested spatial CV) and xgboost | C1: FM ≥ tuned GBDT under spatial shift |
| Florida no_landcover × ring × temporal | same | C1 under temporal shift |
| Kentucky loco | same | C1 at large n, clean morphometric features |
| Minnesota sat_nolc × ring × spatial5 | same | C1 on a third karst region, satellite-only schema |

Statistics: pooled OOF AUC; 5 seeds per model; ΔAUC CI from a bootstrap over spatial blocks (Kentucky: rows) with an
independent seed draw per resample; paired DeLong on the seed-averaged predictions; Holm correction over {3 FMs × 4
settings} per reference. Everything else in `results/` is exploratory or confirmatory of a specific claim (C2–C7) and is
reported in full in `results/README.md`. Output: `results/seeds/summary.md`.

