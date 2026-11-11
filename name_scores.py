"""
Score every labeled name pair with the best combination found by string_matching.py:
    normalized names, mean(Jaro-Winkler, Overlap coefficient on words), threshold 0.773
and write them, highest score first, to results/name_pair_scores.csv.
"""
import os

from string_matching import OUT, load, overlap_coef, prep_norm, sim_jaro_winkler, words

THRESHOLD = 0.773


def main():
    os.makedirs(OUT, exist_ok=True)
    _, _, pairs = load()
    pairs["name_A"] = [prep_norm(v) for v in pairs.x]
    pairs["name_B"] = [prep_norm(v) for v in pairs.y]
    pairs["jaro_winkler"] = [sim_jaro_winkler(a, b) for a, b in zip(pairs.name_A, pairs.name_B)]
    pairs["overlap_word"] = [overlap_coef(words(a), words(b))
                             for a, b in zip(pairs.name_A, pairs.name_B)]
    pairs["score"] = (pairs.jaro_winkler + pairs.overlap_word) / 2
    pairs["predicted"] = (pairs.score >= THRESHOLD).astype(int)
    pairs["correct"] = pairs.predicted == pairs.label
    pairs["identical"] = pairs.name_A == pairs.name_B

    out = (pairs.sort_values(["score", "label"], ascending=False)
           [["split", "ltable_id", "rtable_id", "name_A", "name_B", "jaro_winkler",
             "overlap_word", "score", "label", "predicted", "correct", "identical"]]
           .round({"jaro_winkler": 4, "overlap_word": 4, "score": 4}))
    out.insert(0, "rank", range(1, len(out) + 1))
    path = os.path.join(OUT, "name_pair_scores.csv")
    out.to_csv(path, index=False)

    print(f"Wrote {len(out)} pairs to {path}")
    print(f"Score = 1.0: {(out.score == 1).sum()} pairs "
          f"({out[out.score == 1].label.sum()} true matches)")
    print(f"Predicted matches (score >= {THRESHOLD}): {out.predicted.sum()}, "
          f"correct overall: {out.correct.sum()}/{len(out)}")
    print("\nTop 15 non-identical names:")
    print(out[~out.identical].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
