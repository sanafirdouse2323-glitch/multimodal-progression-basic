# Multimodal disease progression: a basic Python demo

A small, readable Python project written for an application to the PhD in Machine Learning in Medicine (University of Bern). One script runs five experiments, one for each methodological focus area in the project description, using real scikit-learn models and honest evaluation on **simulated** data.

Everything is synthetic: each patient has a hidden disease severity, and imaging, wearable and tabular measurements are noisy views of it. Simulation lets every result be checked against a known truth. Nothing here is a clinical result.

## Run it

```bash
pip install -r requirements.txt
python progression_demo.py     # about 10 seconds; writes results.md and figures.png
pytest                         # 5 tests
```

## What each experiment shows

| Focus area from the posting | Experiment | Result (seed 0) |
|---|---|---|
| Multimodal representation learning | Logistic regression on each modality alone, late fusion and early fusion | Fusion reaches AUC 0.903, against 0.886 for the best single modality and 0.927 for a ceiling that knows the true severity |
| Progression from sparse longitudinal data | Pooled within-patient slope and per-patient offsets from 2 to 6 visits at random times | Estimated rate 0.57 to 0.60 against a true 0.59; recovered offsets correlate 0.96 to 0.99 with the truth |
| Causal ML for treatment effects | Naive difference, regression adjustment and inverse-probability weighting over 200 simulated studies (true effect -0.7) | Naive bias +0.66. Adjusting for a noisy baseline measurement leaves +0.15. An oracle that adjusts for the true severity is unbiased |
| Missing modalities | Whole modalities dropped at random at test time. Boosted trees trained with and without modality dropout, plus a logistic regression baseline | Dropout training helps modestly: AUC 0.866 against 0.846 at 40% missing, and 0.733 against 0.718 at 80% |
| Fairness across subgroups | A minority group (30%) with a 4x noisier wearable sensor. Pooled model, group-reweighted training, separate model per group; 10 seeds | The AUC gap is small (+0.016, SD 0.010). Reweighting narrows it to +0.009 but within noise; separate models do not help |

Full tables are in `results.md` and the plots are in `figures.png`.

## Limitations

- Simulated data and deliberately simple models. The goal is to show the problem settings and a careful evaluation, not state-of-the-art accuracy.
- The missing-modality gain and the fairness mitigation effects are small and come from one simulation setup, so they should not be read as general findings.
- The causal experiment assumes no unmeasured confounding; only noisy measurement of the confounder remains.
- No real clinical, imaging or wearable data are used.

## Next steps

Replace the simulator with a public medical dataset, swap the linear fusion for a learned multimodal encoder (PyTorch), model longitudinal data with a mixed-effects or neural progression model, and use a doubly robust estimator for the causal part.
