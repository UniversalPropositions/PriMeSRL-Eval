import json
import os
import sys

import pandas as pd
import logging

from src.conll.conll2005.span_evaluation import run_span_evaluation
from src.conll.conll2009.head_evaluation import run_head_evaluation
from src.conll.utils import combine_performance_scores
from src.proposed.conversion import transfer_span_info
from src.proposed.comparison import compare_conllu_csv
from src.proposed.evaluation import summarize_srl_comparisons
from src.proposed.utils import output_df
from src.utils import setup_output_folders, download_official_scripts

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def proposed_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder):
    """Evaluate the gold CoNLL-U file against the predicted CoNLL-U file.

    :param gold_conllu_fp: Path to the CoNLL-U file with gold labels.
    :param pred_conllu_fp: Path to the CoNLL-U file with predicted labels.
    :param output_folder: Folder to output the evaluation files and results.
    :return: None
    """
    gold_basename, pred_basename, gold_path, pred_path, out_gold_file, out_pred_file, compare_output_folder = setup_output_folders(gold_conllu_fp, pred_conllu_fp, output_folder)

    logger.info("Running proposed evaluation...")

    logger.info("Converting CoNNL-U file to CSV tables...")
    transfer_span_info(out_gold_file, out_gold_file, compute_span=True)
    transfer_span_info(out_gold_file, out_pred_file, compute_span=True)

    logger.info("Comparing gold SRL with predicted SRL...")
    compare_conllu_csv(gold_path, pred_path, compare_output_folder)

    logger.info("Generating evaluation summary...")
    df_summ = summarize_srl_comparisons(compare_output_folder)

    return df_summ

################################################################################
################################################################################


def official_conll_evaluation(gold_file, pred_file, output_folder):
    """Run the official CoNLL evaluation scripts.

    :param gold_file: Path to the CoNLL file with gold labels.
    :param pred_file: Path to the CoNLL file with predicted labels.
    :param output_folder: Folder to output the evaluation files and results.
    :return: None
    """

    download_official_scripts()

    gold_basename, pred_basename, gold_path, pred_path, out_gold_file, out_pred_file, compare_output_folder = setup_output_folders(gold_file, pred_file, output_folder)

    logger.info("Running CoNLL2009 evaluation...")

    pred_score, sense_score, arg_head_score = run_head_evaluation(out_gold_file, out_pred_file, conllu2x=True, conditional_eval=False)

    rows09 = [
        ['PredicateId'] + list(pred_score),
        ['Predicate'] + list(sense_score),
        ['ArgumentHead'] + list(arg_head_score)
    ]

    colnames = ['Type', 'Correct', 'Spurious', 'Missing', 'Precision', 'Recall', 'F1']
    df_stat09 = pd.DataFrame(rows09, columns=colnames)
    df_stat09.insert(0, column='Metric', value="conll09-head")

    logger.info("Running CoNLL2005 evaluation...")

    _, arg_span_score = run_span_evaluation(out_gold_file, out_pred_file, conllu2x=True)

    rows05 = [
        ['ArgumentSpan'] + list(arg_span_score)
    ]

    df_stat05 = pd.DataFrame(rows05, columns=colnames)
    df_stat05.insert(0, column='Metric', value="conll05-span")

    logger.info("Generating evaluation summary...")

    df_summ = pd.concat([df_stat09, df_stat05]).reset_index()
    _colnames = ['Correct', 'Spurious', 'Missing']
    df_summ[_colnames] = df_summ[_colnames].astype(int)

    out_file = os.path.join(compare_output_folder, 'comparison-results-official-conll.csv')
    output_df(df_summ, out_file)

    return df_summ

