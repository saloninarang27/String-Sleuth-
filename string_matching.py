"""
Home Assignment - Data String Matching (Fodors-Zagats, Magellan repository)

Matching attribute: `name` (restaurant name).

Pipeline
  1. Load labeled pairs (train/valid/test) and look up the `name` of each side.
  2. Preprocess with 3 variants: raw, normalized, normalized + rule-based cleanup.
  3. Score every pair with 13 similarity measures from the lecture
     (sequence-based, set-based, Ratcliff-Obershelp, phonetic).
  4. Build combinations of measures (mean of 2..4 measures; max/min/product for pairs).
  5. For every (preprocessing, combination): pick the threshold that maximizes F1 on
     train+valid, then report precision/recall/F1 on the held-out test set.
  6. Rank and write results + charts to ./results.
"""
import itertools
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "datasets", "Fodors-Zagats")
OUT = os.path.join(HERE, "results")
ATTR = "name"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def prep_raw(s):
    return str(s)


def prep_norm(s):
    """Normalization-based matching: case folding, strip quote/escape artifacts and
    punctuation, collapse whitespace."""
    s = str(s).lower()
    s = s.replace("\\ '", "'").replace("\\'", "'")
    s = re.sub(r"[`'\"]", " ", s)  # dataset wraps values in ` ... ' quotes
    s = re.sub(r"[^a-z0-9& ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


RULES = {"&": "and", "st": "street", "ave": "avenue", "cafe": "cafe", "caffe": "cafe"}
STOP = {"the", "restaurant", "s", "and", "of", "a"}


def prep_rule(s):
    """Rule-based matching on top of normalization: abbreviation/synonym rules,
    drop stop words (e.g. Zagat's 'apple pan the' vs Fodor's 'the apple pan')."""
    toks = [RULES.get(t, t) for t in prep_norm(s).split()]
    toks = [t for t in toks if t not in STOP]
    return " ".join(toks) if toks else prep_norm(s)


PREPS = {"raw": prep_raw, "norm": prep_norm, "norm+rules": prep_rule}


# --------------------------------------------------------------------------
# Sequence-based measures
# --------------------------------------------------------------------------
def levenshtein(x, y):
    n, m = len(x), len(y)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] != y[j - 1]))
        prev = cur
    return prev[m]


def sim_edit(x, y):
    """s(x,y) = 1 - d(x,y) / max(|x|,|y|)"""
    if not x and not y:
        return 1.0
    return 1 - levenshtein(x, y) / max(len(x), len(y))


def sim_needleman_wunsch(x, y, match=1, mismatch=-1, gap=-1):
    """Global alignment score, normalized by the best possible score max(|x|,|y|)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float(n == m)
    prev = [j * gap for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * gap] + [0] * m
        for j in range(1, m + 1):
            s = match if x[i - 1] == y[j - 1] else mismatch
            cur[j] = max(prev[j - 1] + s, prev[j] + gap, cur[j - 1] + gap)
        prev = cur
    return max(0.0, prev[m] / (match * max(n, m)))


def sim_affine_gap(x, y, match=2, mismatch=-1, g_open=-1, g_ext=-0.5):
    """Gotoh affine-gap global alignment: opening a gap costs more than extending it,
    so 'dave smith' vs 'david r. smith' is not punished for the whole inserted block."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float(n == m)
    NEG = -1e9
    M = [[NEG] * (m + 1) for _ in range(n + 1)]
    X = [[NEG] * (m + 1) for _ in range(n + 1)]  # gap in y
    Y = [[NEG] * (m + 1) for _ in range(n + 1)]  # gap in x
    M[0][0] = 0
    for i in range(1, n + 1):
        X[i][0] = g_open + (i - 1) * g_ext
    for j in range(1, m + 1):
        Y[0][j] = g_open + (j - 1) * g_ext
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if x[i - 1] == y[j - 1] else mismatch
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + s
            X[i][j] = max(M[i - 1][j] + g_open, X[i - 1][j] + g_ext)
            Y[i][j] = max(M[i][j - 1] + g_open, Y[i][j - 1] + g_ext)
    best = max(M[n][m], X[n][m], Y[n][m])
    return max(0.0, best / (match * max(n, m)))


def sim_smith_waterman(x, y, match=2, mismatch=-1, gap=-1):
    """Local alignment: best matching substring pair, normalized by match*min(|x|,|y|)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float(n == m)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            s = match if x[i - 1] == y[j - 1] else mismatch
            cur[j] = max(0, prev[j - 1] + s, prev[j] + gap, cur[j - 1] + gap)
            best = max(best, cur[j])
        prev = cur
    return best / (match * min(n, m))


def sim_jaro(x, y):
    """jaro = 1/3 [c/|x| + c/|y| + (c - t/2)/c], window = max(|x|,|y|)/2 - 1."""
    if x == y:
        return 1.0
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    win = max(max(n, m) // 2 - 1, 0)
    xm, ym = [False] * n, [False] * m
    c = 0
    for i in range(n):
        for j in range(max(0, i - win), min(m, i + win + 1)):
            if not ym[j] and x[i] == y[j]:
                xm[i] = ym[j] = True
                c += 1
                break
    if c == 0:
        return 0.0
    xs = [x[i] for i in range(n) if xm[i]]
    ys = [y[j] for j in range(m) if ym[j]]
    t = sum(a != b for a, b in zip(xs, ys))  # t counts mismatched positions; t/2 = transpositions
    return (c / n + c / m + (c - t / 2) / c) / 3


def sim_jaro_winkler(x, y, pw=0.1):
    """(1 - PL*PW) * jaro + PL*PW, PL = common prefix length (capped at 4)."""
    j = sim_jaro(x, y)
    pl = 0
    for a, b in zip(x, y):
        if a != b or pl == 4:
            break
        pl += 1
    return (1 - pl * pw) * j + pl * pw


# --------------------------------------------------------------------------
# Set-based measures
# --------------------------------------------------------------------------
def words(s):
    return set(s.split())


def qgrams(s, q=3):
    s = "#" * (q - 1) + s + "#" * (q - 1)
    return {s[i:i + q] for i in range(len(s) - q + 1)}


def overlap_coef(a, b):
    """|A ∩ B| / min(|A|,|B|)  (overlap count normalized into [0,1])."""
    if not a or not b:
        return float(a == b)
    return len(a & b) / min(len(a), len(b))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dice(a, b):
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


class TfIdf:
    """Cosine similarity of TF-IDF word vectors, idf(t) = log(N / N_t) computed over
    all names in tableA and tableB (the corpus being matched)."""

    def __init__(self, corpus):
        df = Counter()
        for doc in corpus:
            df.update(set(doc.split()))
        self.n = len(corpus)
        self.idf = {t: math.log(self.n / c) for t, c in df.items()}

    def vec(self, s):
        tf = Counter(s.split())
        return {t: c * self.idf.get(t, math.log(self.n)) for t, c in tf.items()}

    def sim(self, x, y):
        if x == y:
            return 1.0
        a, b = self.vec(x), self.vec(y)
        dot = sum(v * b.get(t, 0) for t, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------
# Ratcliff-Obershelp and Soundex
# --------------------------------------------------------------------------
def _ro_common(x, y):
    """Characters in common: longest common substring, then recurse on left & right."""
    if not x or not y:
        return 0
    best, bi, bj = 0, 0, 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, bi, bj = cur[j], i, j
        prev = cur
    if best == 0:
        return 0
    return best + _ro_common(x[:bi - best], y[:bj - best]) + _ro_common(x[bi:], y[bj:])


def sim_ratcliff(x, y):
    if not x and not y:
        return 1.0
    return 2 * _ro_common(x, y) / (len(x) + len(y))


SDX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
       **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(w):
    """Slide algorithm: keep 1st letter; drop w/h; map to digits; collapse runs;
    drop non-digits; pad with 0 to 4 chars."""
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return ""
    rest = [c for c in w[1:] if c not in "wh"]
    codes = [SDX.get(c, c) for c in rest]
    collapsed = []
    for c in codes:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    digits = [c for c in collapsed if c.isdigit()]
    return (w[0].upper() + "".join(digits) + "000")[:4]


def sim_soundex(x, y):
    """Names are multi-word, so compare the Jaccard of per-token Soundex codes."""
    a = {soundex(t) for t in x.split()} - {""}
    b = {soundex(t) for t in y.split()} - {""}
    return jaccard(a, b)


# --------------------------------------------------------------------------
# Measure registry
# --------------------------------------------------------------------------
def build_measures(tfidf):
    return {
        "Levenshtein": sim_edit,
        "Needleman-Wunsch": sim_needleman_wunsch,
        "Affine gap": sim_affine_gap,
        "Smith-Waterman": sim_smith_waterman,
        "Jaro": sim_jaro,
        "Jaro-Winkler": sim_jaro_winkler,
        "Overlap (word)": lambda x, y: overlap_coef(words(x), words(y)),
        "Jaccard (word)": lambda x, y: jaccard(words(x), words(y)),
        "Jaccard (3-gram)": lambda x, y: jaccard(qgrams(x), qgrams(y)),
        "Dice (3-gram)": lambda x, y: dice(qgrams(x), qgrams(y)),
        "TF/IDF cosine": tfidf.sim,
        "Ratcliff-Obershelp": sim_ratcliff,
        "Soundex": sim_soundex,
    }


FAMILY = {
    "Levenshtein": "Sequence", "Needleman-Wunsch": "Sequence", "Affine gap": "Sequence",
    "Smith-Waterman": "Sequence", "Jaro": "Sequence", "Jaro-Winkler": "Sequence",
    "Overlap (word)": "Set", "Jaccard (word)": "Set", "Jaccard (3-gram)": "Set",
    "Dice (3-gram)": "Set", "TF/IDF cosine": "Set",
    "Ratcliff-Obershelp": "Similarity", "Soundex": "Phonetic",
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def prf(scores, labels, t):
    pred = scores >= t
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(scores, labels):
    """Threshold maximizing F1 (predict match iff score >= t); among ties the lowest t.
    Vectorized: sort descending, F1 at every distinct cutoff via cumulative sums."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    last = np.r_[s[1:] != s[:-1], True]  # last index of each run of equal scores
    tp, fp, s = tp[last], fp[last], s[last]
    f1 = 2 * tp / (tp + fp + (y == 1).sum())
    i = int(np.argmax(f1))
    return float(s[i]), float(f1[i])


def cv_f1(scores, labels, folds):
    """Cross-validated F1: tune threshold on k-1 folds, apply to held-out fold,
    pool predictions over all folds."""
    pred = np.zeros(len(scores), bool)
    for k in np.unique(folds):
        tr = folds != k
        t, _ = best_threshold(scores[tr], labels[tr])
        pred[~tr] = scores[~tr] >= t
    tp = np.sum(pred & (labels == 1))
    return 2 * tp / (pred.sum() + (labels == 1).sum())


def auc(scores, labels):
    """ROC-AUC via the rank-sum (Mann-Whitney) formula, ties averaged."""
    order = pd.Series(scores).rank(method="average").to_numpy()
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (order[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load():
    A = pd.read_csv(os.path.join(DATA, "tableA.csv"))
    B = pd.read_csv(os.path.join(DATA, "tableB.csv"))
    frames = []
    for split in ["train", "valid", "test"]:
        d = pd.read_csv(os.path.join(DATA, f"{split}.csv"))
        d["split"] = split
        frames.append(d)
    pairs = pd.concat(frames, ignore_index=True)
    pairs["x"] = A.set_index("id").loc[pairs.ltable_id, ATTR].to_numpy()
    pairs["y"] = B.set_index("id").loc[pairs.rtable_id, ATTR].to_numpy()
    return A, B, pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    A, B, pairs = load()
    labels = pairs.label.to_numpy()
    tune = (pairs.split != "test").to_numpy()
    test = (pairs.split == "test").to_numpy()
    # stratified 5-fold assignment over all labeled pairs (fixed seed)
    rng = np.random.default_rng(42)
    folds = np.zeros(len(labels), int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(labels == c))
        folds[idx] = np.arange(len(idx)) % 5

    # 1) individual measure scores for every preprocessing variant
    S = {}  # (prep, measure) -> np.array of scores
    for pname, pf in PREPS.items():
        corpus = [pf(v) for v in pd.concat([A[ATTR], B[ATTR]])]
        measures = build_measures(TfIdf(corpus))
        xs = [pf(v) for v in pairs.x]
        ys = [pf(v) for v in pairs.y]
        for mname, fn in measures.items():
            S[(pname, mname)] = np.array([fn(a, b) for a, b in zip(xs, ys)])
        pairs[f"{pname}_x"], pairs[f"{pname}_y"] = xs, ys
    mnames = list(build_measures(TfIdf(["a"])).keys())

    def evaluate(pname, combo, fusion, scores):
        t, f_tune = best_threshold(scores[tune], labels[tune])
        p, r, f = prf(scores[test], labels[test], t)
        return {
            "preprocessing": pname, "combination": " + ".join(combo), "k": len(combo),
            "fusion": fusion, "threshold": round(t, 4), "F1_tune": round(f_tune, 4),
            "P_test": round(p, 4), "R_test": round(r, 4), "F1_test": round(f, 4),
            "AUC_test": round(auc(scores[test], labels[test]), 4),
            "F1_cv5": round(cv_f1(scores, labels, folds), 4),
        }

    rows = []
    fusions = {
        "mean": lambda M: M.mean(axis=0),
        "max": lambda M: M.max(axis=0),
        "min": lambda M: M.min(axis=0),
        "product": lambda M: M.prod(axis=0),
    }
    for pname in PREPS:
        for k in range(1, 5):
            for combo in itertools.combinations(mnames, k):
                M = np.vstack([S[(pname, m)] for m in combo])
                use = ["mean"] if k == 1 else (fusions if k == 2 else ["mean"])
                for fu in use:
                    rows.append(evaluate(pname, combo, fu, fusions[fu](M)))
    res = pd.DataFrame(rows)
    # Rank on the tuning split (train+valid); test F1 / AUC break ties. Test is never
    # used to choose the threshold.
    res = res.sort_values(["F1_tune", "F1_cv5", "F1_test", "k"],
                          ascending=[False, False, False, True]).reset_index(drop=True)
    res.to_csv(os.path.join(OUT, "all_combinations.csv"), index=False)

    single = res[res.k == 1].sort_values(["preprocessing", "F1_tune"], ascending=[True, False])
    single.to_csv(os.path.join(OUT, "single_measures.csv"), index=False)

    pairs_out = pairs[["split", "ltable_id", "rtable_id", "label", "x", "y"]].copy()
    for (pname, m), v in S.items():
        if pname == "norm+rules":
            pairs_out[m] = np.round(v, 4)
    pairs_out.to_csv(os.path.join(OUT, "pair_scores_norm_rules.csv"), index=False)

    # Error analysis for the overall best combination
    best = res.iloc[0]
    combo = best.combination.split(" + ")
    M = np.vstack([S[(best.preprocessing, m)] for m in combo])
    sc = fusions[best.fusion](M)
    pred = sc >= best.threshold
    err = pairs.loc[test & (pred != (labels == 1)), ["ltable_id", "rtable_id", "label", "x", "y"]].copy()
    err["score"] = np.round(sc[test & (pred != (labels == 1))], 4)
    err.to_csv(os.path.join(OUT, "best_errors_test.csv"), index=False)

    # Hardest cases on all labeled pairs (for discussion): matches with low raw
    # Levenshtein and non-matches with high raw Levenshtein
    lev = S[("raw", "Levenshtein")]
    hard = pairs.assign(lev_raw=np.round(lev, 3))
    hard_pos = hard[hard.label == 1].nsmallest(8, "lev_raw")[["x", "y", "lev_raw"]]
    hard_neg = hard[hard.label == 0].nlargest(8, "lev_raw")[["x", "y", "lev_raw"]]
    hard_pos.to_csv(os.path.join(OUT, "hard_matches.csv"), index=False)
    hard_neg.to_csv(os.path.join(OUT, "hard_nonmatches.csv"), index=False)

    make_charts(res, S, labels, test, mnames)
    print_summary(res, pairs, err)


def make_charts(res, S, labels, test, mnames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preps = list(PREPS)
    colors = {"raw": "#9aa5b1", "norm": "#4c8bf5", "norm+rules": "#1b4f9c"}

    # Chart 1: single measures, test F1 per preprocessing
    s = res[res.k == 1]
    order = (s[s.preprocessing == "norm+rules"].sort_values("F1_test")
             .combination.tolist())
    fig, ax = plt.subplots(figsize=(9, 6.5))
    h = 0.27
    for i, p in enumerate(preps):
        vals = s[s.preprocessing == p].set_index("combination").loc[order, "F1_test"]
        ax.barh(np.arange(len(order)) + (i - 1) * h, vals, h, label=p, color=colors[p])
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_xlabel("Test F1 (threshold tuned on train+valid)")
    ax.set_xlim(0, 1.02)
    ax.set_title("Single measures on `name` — effect of preprocessing")
    ax.legend(title="Preprocessing", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_single_measures.png"), dpi=150)
    plt.close(fig)

    # Chart 2: best F1 by combination size (k): tuning F1 and 5-fold CV F1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, ttl in [(axes[0], "F1_tune", "Best F1 on train+valid (tuning)"),
                         (axes[1], "F1_cv5", "Best 5-fold cross-validated F1")]:
        g = res.groupby(["preprocessing", "k"])[col].max().unstack(0)[preps]
        for p in preps:
            ax.plot(g.index, g[p], marker="o", label=p, color=colors[p], lw=2)
        ax.set_xticks(g.index)
        ax.set_xlabel("Number of measures combined (k)")
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1")
    axes[1].legend(title="Preprocessing", loc="lower right")
    fig.suptitle("Does combining measures help?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_combination_size.png"), dpi=150)
    plt.close(fig)

    # Chart 3: score distributions for match vs non-match (norm+rules, test split)
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), sharey=False)
    for ax, m in zip(axes.flat, mnames):
        v = S[("norm+rules", m)]
        ax.hist(v[labels == 0], bins=20, range=(0, 1), alpha=0.6, color="#9aa5b1", label="non-match")
        ax.hist(v[labels == 1], bins=20, range=(0, 1), alpha=0.8, color="#d9480f", label="match")
        ax.set_title(m, fontsize=10)
        ax.set_yscale("log")
    for ax in axes.flat[len(mnames):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Score distributions, all labeled pairs (norm+rules); log count", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_distributions.png"), dpi=130)
    plt.close(fig)

    # Chart 4: pairwise mean-fusion heatmap (norm+rules, test F1)
    pr = res[(res.k == 2) & (res.fusion == "mean") & (res.preprocessing == "norm+rules")]
    sg = res[(res.k == 1) & (res.preprocessing == "norm+rules")].set_index("combination").F1_test
    H = pd.DataFrame(np.nan, index=mnames, columns=mnames)
    for _, r in pr.iterrows():
        a, b = r.combination.split(" + ")
        H.loc[a, b] = H.loc[b, a] = r.F1_test
    for m in mnames:
        H.loc[m, m] = sg[m]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(H.to_numpy(), cmap="Blues", vmin=H.min().min(), vmax=H.max().max())
    ax.set_xticks(range(len(mnames)), mnames, rotation=60, ha="right")
    ax.set_yticks(range(len(mnames)), mnames)
    for i in range(len(mnames)):
        for j in range(len(mnames)):
            v = H.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > (H.min().min() + H.max().max()) / 2 else "black")
    fig.colorbar(im, ax=ax, label="Test F1")
    ax.set_title("Pairwise mean-fusion test F1 (norm+rules); diagonal = single measure")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_pair_heatmap.png"), dpi=150)
    plt.close(fig)


def print_summary(res, pairs, err):
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    cols = ["preprocessing", "combination", "fusion", "threshold", "F1_tune",
            "P_test", "R_test", "F1_test", "AUC_test", "F1_cv5"]
    print(f"Pairs: {len(pairs)}  matches: {pairs.label.sum()}  "
          f"combinations evaluated: {len(res)}")
    print("\n=== Single measures (ranked by tune F1) per preprocessing ===")
    for p in PREPS:
        print(f"\n-- {p} --")
        print(res[(res.k == 1) & (res.preprocessing == p)][cols].to_string(index=False))
    print("\n=== Top 20 combinations overall ===")
    print(res[cols].head(20).to_string(index=False))
    print("\n=== Best per k ===")
    print(res.groupby("k").head(1)[cols + ["k"]].to_string(index=False))
    print("\n=== Best per fusion (k=2) ===")
    print(res[res.k == 2].groupby("fusion").head(1)[cols].to_string(index=False))
    print("\n=== Best per preprocessing ===")
    print(res.groupby("preprocessing").head(1)[cols].to_string(index=False))
    print("\n=== Top 10 by 5-fold CV F1 ===")
    print(res.sort_values(["F1_cv5", "F1_tune"], ascending=False)[cols].head(10).to_string(index=False))
    print("\n=== Test errors of best combination ===")
    print(err.to_string(index=False))


if __name__ == "__main__":
    main()
