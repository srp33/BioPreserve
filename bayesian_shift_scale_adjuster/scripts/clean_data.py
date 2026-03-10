import argparse
import polars as pl

def clean_data(train_path, test_path, output_train, output_test, cols_to_drop):

    """Clean the specified data files by handling nulls and ensuring that each file has the same genes."""

    gse = pl.read_csv(train_path).drop(cols_to_drop)
    metabric = pl.read_csv(test_path).drop(cols_to_drop)

    # Report and handle nulls
    for name, df in [("train (GSE)", gse), ("test (METABRIC)", metabric)]:

        # Locate the nulls
        null_counts = df.null_count().transpose(include_header=True, column_names=["nulls"])
        cols_with_nulls = null_counts.filter(pl.col("nulls") > 0)

        # Report the found nulls
        if cols_with_nulls.is_empty():
            print(f"{name}: no null values found")
        else:
            print(f"{name}: null counts per column:")
            print(cols_with_nulls)

    # Drop columns that are entirely null in either dataset (unmeasured genes)

    # Check for the existence of null columns
    n_rows_gse = gse.height
    n_rows_metabric = metabric.height
    all_null_gse = {col for col in gse.columns if gse[col].null_count() == n_rows_gse}
    all_null_metabric = {col for col in metabric.columns if metabric[col].null_count() == n_rows_metabric}
    cols_to_drop_null = all_null_gse | all_null_metabric

    # Drop null columns if they exist
    if cols_to_drop_null:
        print(f"\nDropping {len(cols_to_drop_null)} entirely-null column(s): {sorted(cols_to_drop_null)}")
        gse = gse.drop([c for c in cols_to_drop_null if c in gse.columns])
        metabric = metabric.drop([c for c in cols_to_drop_null if c in metabric.columns])

    # Fill all nulls with 0
    # Note: this is done based off of the assumption that there are relatively few. nulls, more advanced null handling should be done otherwise
    gse_clean = gse.fill_null(0)
    metabric_clean = metabric.fill_null(0)

    gse_clean.write_csv(output_train)
    metabric_clean.write_csv(output_test)

    return f"SUCCESS: train={gse_clean.shape}, test={metabric_clean.shape}"


def main():

    # Handle all relevant arguments

    parser = argparse.ArgumentParser(description="Clean the specified data files by handling nulls and ensuring that each file has the same genes.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--output-train", required=True, help="Path for cleaned train CSV.")
    parser.add_argument("--output-test", required=True, help="Path for cleaned test CSV.")
    parser.add_argument("--cols-to-drop", nargs="+", default=[], help="Column names to drop.")
    args = parser.parse_args()

    message = clean_data(args.train, args.test, args.output_train, args.output_test, args.cols_to_drop)
    print(message, flush=True)


if __name__ == "__main__":

    main()