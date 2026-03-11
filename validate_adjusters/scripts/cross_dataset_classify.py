import argparse

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import matthews_corrcoef, r2_score


def cross_dataset_classify(X_train_path, X_test_path, adjuster_paths,
                            continuous_metadata, output_path):
    X_train_df = pl.read_csv(X_train_path)
    X_test_df = pl.read_csv(X_test_path)

    results = []

    for adjuster_name, paths in adjuster_paths.items():
        y_train_df = pl.read_csv(paths["train"])
        y_test_adj_df = pl.read_csv(paths["test"])

        # Use only numeric gene columns present in both datasets (exclude meta_* columns)
        numeric_train = set(y_train_df.select(cs.numeric()).columns) - set(y_train_df.select(cs.starts_with("meta")).columns)
        numeric_test = set(y_test_adj_df.select(cs.numeric()).columns) - set(y_test_adj_df.select(cs.starts_with("meta")).columns)
        common_genes = [g for g in y_train_df.columns if g in numeric_train and g in numeric_test]
        X_genes_train_full = y_train_df.select(common_genes).to_numpy()
        X_genes_test_full = y_test_adj_df.select(common_genes).to_numpy()

        for meta_col in X_train_df.columns:
            label_train_series = X_train_df[meta_col]
            label_test_series = X_test_df[meta_col]

            valid_train = label_train_series.is_not_null().to_numpy()
            valid_test = label_test_series.is_not_null().to_numpy()

            y_train_label = label_train_series.drop_nulls().to_numpy()
            y_test_label = label_test_series.drop_nulls().to_numpy()

            X_genes_train = X_genes_train_full[valid_train]
            X_genes_test = X_genes_test_full[valid_test]

            is_continuous = meta_col in continuous_metadata

            if is_continuous:
                metric = "R2"
                reg_fwd = HistGradientBoostingRegressor()
                reg_fwd.fit(X_genes_train, y_train_label)
                score_fwd = r2_score(y_test_label, reg_fwd.predict(X_genes_test))

                reg_rev = HistGradientBoostingRegressor()
                reg_rev.fit(X_genes_test, y_test_label)
                score_rev = r2_score(y_train_label, reg_rev.predict(X_genes_train))
            else:
                metric = "MCC"
                # Skip if either dataset has only one class
                if len(np.unique(y_train_label)) < 2 or len(np.unique(y_test_label)) < 2:
                    results.append({
                        "adjuster": adjuster_name,
                        "metadata_label": meta_col,
                        "metric": metric,
                        "score_gse_to_metabric": float("nan"),
                        "score_metabric_to_gse": float("nan"),
                        "score_mean": float("nan"),
                    })
                    continue

                clf_fwd = HistGradientBoostingClassifier()
                clf_fwd.fit(X_genes_train, y_train_label)
                score_fwd = matthews_corrcoef(y_test_label, clf_fwd.predict(X_genes_test))

                clf_rev = HistGradientBoostingClassifier()
                clf_rev.fit(X_genes_test, y_test_label)
                score_rev = matthews_corrcoef(y_train_label, clf_rev.predict(X_genes_train))

            results.append({
                "adjuster": adjuster_name,
                "metadata_label": meta_col,
                "metric": metric,
                "score_gse_to_metabric": score_fwd,
                "score_metabric_to_gse": score_rev,
                "score_mean": (score_fwd + score_rev) / 2,
            })

    pl.DataFrame(results).write_csv(output_path)

    return f"SUCCESS: {len(results)} classification results written"


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset classification to evaluate gene expression adjusters.")
    parser.add_argument("--X-train", required=True, help="Path to train metadata features CSV.")
    parser.add_argument("--X-test", required=True, help="Path to test metadata features CSV.")
    parser.add_argument("--adjusters", nargs="+", required=True,
                        help="Adjuster name=train_path:test_path triplets.")
    parser.add_argument("--continuous-metadata", nargs="*", default=[],
                        help="Metadata columns to treat as continuous (use R2 instead of MCC).")
    parser.add_argument("--output", required=True, help="Output path for classification results CSV.")
    args = parser.parse_args()

    adjuster_paths = {}
    for pair in args.adjusters:
        name, paths = pair.split("=", 1)
        train_path, test_path = paths.split(":", 1)
        adjuster_paths[name] = {"train": train_path, "test": test_path}

    message = cross_dataset_classify(
        args.X_train, args.X_test,
        adjuster_paths, args.continuous_metadata, args.output,
    )
    print(message, flush=True)


if __name__ == "__main__":
    main()
