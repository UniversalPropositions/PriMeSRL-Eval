import os
from subprocess import check_output

from src.conll.conllu_conversion import read_file, get_conllu_sen, conllu_conversion
from src.conll.utils import check_predicate_pos
import logging

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def official_conll2005_evaluation_script(gold_file, pred_file):
    """Run the official CoNLL2005 evaluation script and parse the output.

    :param gold_file: Gold file in CoNLL2005 format.
    :param pred_file: Predicted file in CoNLL2005 format.
    :return: P R F numbers for argument span.
    """
    script_dir = os.path.dirname(__file__)
    args = ['perl', os.path.join(script_dir, 'srl-eval.pl'), gold_file, pred_file]
    with open(os.devnull, 'w') as devnull:
        output = check_output(args, stderr=devnull)
        output = output.decode('utf-8')

    lines = output.split('\n')
    for line in lines:
        if line.find("Overall") > 0:
            overall = line
            break

    logger.debug("srl-eval.pl | Output:\n%s", output)
    logger.debug("srl-eval.pl | End output --------------------")

    vals = [float(v) for v in overall.strip().split()[1:]]
    correct, spurious, missing, precision, recall, f1 = vals

    logger.debug("Precision: %.2f | Recall: %.2f | F1: %.2f", precision, recall, f1)

    # Normalize the output to be consistent with the others.
    return correct, spurious, missing, precision/100, recall/100, f1/100


def run_span_evaluation(gold_file, pred_file, conllu2x):
    """Wrapper for the CoNLL2005 evaluation that evaluates only the span of the
    arguments.

    :param gold_file: Gold file.
    :param pred_file: Predicted file.
    :param conllu2x: bool: True to convert from CoNLL-U to CoNLL2005 format.
    :return: Tuple of sets of scores for the Predicate, Predicate Sense, and Argument.
    """
    gold_file_conll_filename = ".".join(gold_file.split(".")[:-1]) + ".span.gold"
    gold_data = read_file(gold_file)
    gold_sen = get_conllu_sen(gold_data)

    predfile_conll_filename = ".".join(pred_file.split(".")[:-1]) + ".span.pred"
    pred_data = read_file(pred_file)
    pred_sen = get_conllu_sen(pred_data)

    if conllu2x:
        gold_sen, pred_sen, pred_stat, argument_count = check_predicate_pos(gold_sen[0], pred_sen[0])

    if conllu2x:
        # Convert from CoNLL-U to CoNLL2005 format.
        logger.debug("Supposing both %s and %s files are in CoNLL-U format.", gold_file, pred_file)
        logger.debug("Converting %s from CoNLL-U format to CoNLL2005 format.", gold_file)
        gold_filename = conllu_conversion(gold_sen, gold_file_conll_filename, conversion='05')
        logger.debug("Converting %s from CoNLL-U format to CoNLL2005 format ", pred_file)
        pred_filename = conllu_conversion(pred_sen, predfile_conll_filename, conversion='05')
        logger.debug("*****Evaluated only on common predicates in both the files*****")

        arg_span_stat = official_conll2005_evaluation_script(gold_filename, pred_filename)
    else:
        arg_span_stat = official_conll2005_evaluation_script(gold_file, pred_file)

    return pred_stat, arg_span_stat
