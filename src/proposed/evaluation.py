import os
import json
import pandas as pd
import logging
from collections import Counter

from src.proposed.utils import output_df

logger = logging.getLogger(__name__)

# List all the labels for packing into a summary dataframe, in this order.
all_labels = ['Correct', 'Incorrect', 'Missing', 'Spurious', 'SameValueDiffSpan', 'SameSpanDiffValue',
              'SameHeadSameValueDiffSpan', 'SameValueSameSpanDiffHead']

# Column names for the summary dataframe.
cols = ['Metric', 'Type', *all_labels, 'Precision', 'Recall', 'F1', 'Accuracy']


################################################################################
################################################################################


def error_counts_to_tp_fp_tn_fn(cc, val):
    """Count the labels for TP FP TN FN values as given in metrics.json.

    :param cc: Class counts as a dict.
    :param val: Labels to be combined as TP FP TN FN numbers.
    :return: Counts of TP FP TN FN values.
    """
    counts = {}
    for k in val:
        total = 0
        for v in val[k]:
            if v in cc:
                total += cc[v]
        counts[k] = total
    return counts


def calculate_prfa(counts):
    """Calculate the precision, recall, F1, and accuracy from the counts of
    TP FP TN FN values.

    :param counts: TP FP TN FN values.
    :return: Precision, recall, F1, and accuracy values.
    """

    # FP and TN values may be zero because function generating count may not
    # generate all values if they do not exist.
    for v in ['tp', 'fp', 'tn', 'fn']:
        if v not in counts:
            counts[v] = 0.0

    if counts["tp"] + counts["fp"] == 0.0:
        precision = 0.0
    else:
        precision = counts["tp"] / (0. + counts["tp"] + counts["fp"])

    if counts["tp"] + counts["fn"] == 0.0:
        recall = 0.0
    else:
        recall = counts["tp"] / (0. + counts["tp"] + counts["fn"])

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (0. + precision + recall)

    if counts['tp'] + counts['fp'] + counts['tn'] + counts['fn'] == 0.0:
        acc = 0.0
    else:
        acc = (counts['tp'] + counts['tn']) / (0. + counts['tp'] + counts['fp'] + counts['tn'] + counts['fn'])
    return precision, recall, f1, acc


def merge_counts(counts1, counts2):
    """Merge two dictionaries with value counts.

    :param counts1: First dict of value counts.
    :param counts2: Second dict of value counts.
    :return: Dict of sum of value counts.
    """
    counts3 = Counter(counts1)
    counts3.update(counts2)
    return counts3


def counts_to_row(cc):
    """Flatten the class/label count (cc) to a row for packing into a dataframe.

    :param cc: Dict of class counts.
    :return: List of values in the order given by `all_labels`.
    """
    ccc = []
    for label in all_labels:
        if label in cc:
            ccc.append(cc[label])
        else:
            ccc.append(0)
    return ccc


################################################################################
################################################################################

def summarize_srl_comparisons(compare_folder):
    """Summarize the SRL comparisons of the transformed conllu to csv files.

    Expects the following files in the comparison output folder:
    CompPredId.csv, CompPredSense.csv, CompRoleArgIdLabel.csv, CompContextArgIdLabel.csv
    The comparison label output field in these files will be summarized into
    a dataframe containing different metrics.

    :param compare_folder: Folder containing the comparison outputs.
    :return: Dataframe of the summary of the comparison results.
    """

    with open(os.path.join('conf', 'metrics.json')) as f:
        metrics = json.load(f)

    comp_file = os.path.join(compare_folder, 'CompPredId.csv')
    df_pred = pd.read_csv(comp_file)
    predCC = df_pred['comparison_label_auto'].value_counts().to_dict()

    comp_file = os.path.join(compare_folder, 'CompPredSense.csv')
    df_sense = pd.read_csv(comp_file)
    senseCC = df_sense['comparison_label_auto'].value_counts().to_dict()

    comp_file = os.path.join(compare_folder, 'CompRoleArgIdLabel.csv')
    try:
        df_role = pd.read_csv(comp_file)
        roleCC = df_role['comparison_label_auto'].value_counts().to_dict()
    except:
        roleCC = {}

    comp_file = os.path.join(compare_folder, 'CompContextArgIdLabel.csv')
    try:
        df_context = pd.read_csv(comp_file)
        contextCC = df_context['comparison_label_auto'].value_counts().to_dict()
    except:
        contextCC = {}

    # Calculate the metrics based on how the TP FP TN FN are defined for
    # each different set of evaluation metrics.
    output = []
    for metric in metrics:
        preds = error_counts_to_tp_fp_tn_fn(predCC, metrics[metric])
        predcc = counts_to_row(predCC)

        senses = error_counts_to_tp_fp_tn_fn(senseCC, metrics[metric])
        sensecc = counts_to_row(senseCC)

        roles = error_counts_to_tp_fp_tn_fn(roleCC, metrics[metric])
        rolecc = counts_to_row(roleCC)

        contexts = error_counts_to_tp_fp_tn_fn(contextCC, metrics[metric])
        contextcc = counts_to_row(contextCC)

        all = merge_counts(roles, contexts)
        allcc = [sum(x) for x in zip(rolecc, contextcc)]

        # Multiply the non-head values by the predicate sense accuracy.
        p, r, f, a = calculate_prfa(senses)
        v = [*calculate_prfa(preds)]
        v = [a * x for x in v]

        # Pack these values into a final results dataframe with the counts of
        # the different classes.
        output.append([metric, 'PredicateId', *predcc, *calculate_prfa(preds)])
        output.append([metric, 'Predicate'] + predcc + v)
        output.append([metric, 'ArgumentHead', *allcc, *calculate_prfa(all)])
        output.append([metric, 'CoreArgHead', *rolecc, *calculate_prfa(roles)])
        output.append([metric, 'ContextArgHead', *contextcc, *calculate_prfa(contexts)])

    # Pack these metrics together into one large dataframe.
    # Different comparisons will have different counts for each label, so there
    # will be many zero entries depending on the evaluation metric.
    # We do this to have one dataframe of results that we can process further.
    df = pd.DataFrame(output, columns=cols)
    out_file = os.path.join(compare_folder, 'comparison-results-proposed.csv')
    output_df(df, out_file)

    return df
