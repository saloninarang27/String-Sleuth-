# NameMatch Lab

## When restaurant names get messy, similarity does the detective work

A reproducible Python study of string matching for linking restaurant records
from the Fodor's and Zagat guides in the Fodors-Zagats dataset.

The project compares **13 similarity measures** across raw, normalized, and
rule-enhanced names. It evaluates **3,978 measure/preprocessing combinations**
using train/validation threshold tuning, held-out test metrics, and 5-fold
cross-validation.

### Headline result

The strongest simple approach is:

> Normalize the names, then take the mean of Jaro-Winkler and word overlap,
> using a match threshold of `0.773`.

It reaches **0.951 5-fold cross-validated F1** and **0.913 test F1**. The
remaining errors show why name-only matching has a ceiling: different
restaurants can share the same or nearly identical name. Address, phone, and
city would be needed for a production-quality linker.

## Run it

Requires Python 3.11+ with `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```powershell
python string_matching.py
python name_scores.py
```

The scripts write detailed CSV metrics and figures to `results/`. The full
analysis and interpretation are in [REPORT.md](REPORT.md).

## Project layout

- `string_matching.py` - similarity implementations, evaluation, and charts
- `name_scores.py` - scores all labeled pairs with the selected method
- `datasets/Fodors-Zagats/` - source tables and labeled train/valid/test pairs
- `results/` - generated metrics, error analysis, and figures
- `REPORT.md` - detailed methodology, findings, and limitations
