import argparse
import pandas as pd
from src.utils import format_dataframe

################################################################################
################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format the comparison results.')
    parser.add_argument('-c', '--csv-file', type=str, default="tmp/compare-conll09.brown.test-conll09.brown.pred.roberta-base/comparison-results-proposed.csv", help='Comparison results.')
    args = parser.parse_args()

    csv_file = args.csv_file

    cols = ['Metric', 'Type', 'Precision', 'Recall', 'F1']

    df = pd.read_csv(csv_file)

    print(format_dataframe(df[cols]))
