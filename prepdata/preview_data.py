import pandas as pd
import sys

def preview_tsv(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        preview = df.iloc[:10, :5]
        print(preview)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.ParserError:
        print("Error parsing the TSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_geo_tsv.py <file_path>")
    else:
        preview_tsv(sys.argv[1])