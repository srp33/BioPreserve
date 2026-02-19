import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("expression_file", help="Path to the expression file")
parser.add_argument("clinical_file", help="Path to the clinical file")
parser.add_argument("outfile", help="Path to the output file")
args = parser.parse_args()

unadj = pd.read_csv(args.expression_file, sep="\t")\
    .rename(columns={"Unnamed: 0": "Sample"})
clinical = pd.read_csv(args.clinical_file, index_col=False, sep="\t")

tot = pd.concat([
    clinical.set_index("SampleID"),
    unadj.set_index("Sample")
], axis="columns", join="inner")\
    .reset_index()\
    .rename(columns={"index": "Sample"})

tot.to_csv(args.outfile, index=False)
