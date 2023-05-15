import os
from subprocess import check_output
from src.conll.conllu_conversion import read_file, conllu_conversion, get_conllu_sen
from src.conll.utils import check_predicate_pos
import logging

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def official_conll09_evaluation_script(gold_file, pred_file, argument_count, syn=False, conditional_eval=False):
    """Run the official CoNLL2009 evaluation script and parse the output.

    TODO: Explain the conditional argument evaluation.

    :param gold_file: Gold file in CoNLL2009 format.
    :param pred_file: Predicted file in CoNLL2009 format.
    :param argument_count: Count of spurious and missing arguments.
    :param syn: Whether to evaluate syntactic dependency.
    :param conditional_eval: bool; default: False; Whether perform conditional argument evaluation.
    :return: P R F numbers for Predicate Sense and Argument Head.
    """
    script_dir = os.path.dirname(__file__)
    args = ['perl', os.path.join(script_dir, 'eval09.pl'), '-g', gold_file, '-s', pred_file, '-q']
    with open(os.devnull, 'w') as devnull:
        output = check_output(args, stderr=devnull)
        output = output.decode('utf-8')

    # Just want to return labeled and unlabeled semantic F1 scores
    lines = output.split('\n')
    lf1_line = [line for line in lines if line.startswith('  Labeled F1')][0]
    labeled_f1 = float(lf1_line.strip().split(' ')[-1])
    uf1_line = [line for line in lines if line.startswith('  Unlabeled F1')][0]
    unlabeled_f1 = float(uf1_line.strip().split(' ')[-1])

    if not conditional_eval:
        argument_count['missing'] = 0
        argument_count["spurious"] = 0

    logger.debug("eval09.pl | Output:\n%s", output)
    logger.debug("eval09.pl | End output --------------------")

    if not syn:
        arg_P = (int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[0].split("(")[-1])) / (
                            int(
                                [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[
                                    1].split("(")[-1]) + argument_count["spurious"])
        sens_P = int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[1].split(")")[0]) / int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[2].split(")")[0])

        arg_R = (int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[0].split("(")[-1])) / (int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[1].split("(")[-1]) +
                                                                                                                  argument_count[
                                                                                                                      "missing"])
        sens_R = int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[1].split(")")[0]) / int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[2].split(")")[0])
        arg_f1 = (2 * arg_P * arg_R) / (arg_P + arg_R + 0.0000001)
        sens_f1 = (2 * sens_P * sens_R) / (sens_P + sens_R + 0.0000001)

        logger.debug("Argument p: %.2f, r: %.2f, f1: %.2f", arg_P, arg_R, arg_f1)
        logger.debug("Sense p: %.2f, r: %.2f, f1: %.2f", sens_P, sens_R, sens_f1)
        return (0, 0, 0, arg_P, arg_R, arg_f1), (0, 0, 0, sens_P, sens_R, sens_f1)
    else:
        uf1_line = [line for line in lines if line.startswith('  Labeled   attachment score:')][0]
        LAS = float(uf1_line.strip().split(' ')[-2])
        uf1_line = [line for line in lines if line.startswith('  Unlabeled attachment score:')][0]
        UAS = float(uf1_line.strip().split(' ')[-2])

        logger.debug("LAS: %.2f | UAS: %.2f", LAS, UAS)
        return LAS, UAS


def run_head_evaluation(gold_file, pred_file, conllu2x, conditional_eval=False):
    """Wrapper for the CoNLL2009 evaluation that evaluates only the head of the
    arguments.

    :param gold_file: Gold file.
    :param pred_file: Predicted file.
    :param conllu2x: bool: True to convert from CoNLL-U to CoNLL2009 format.
    :param conditional_eval: bool; default: False; Whether perform conditional argument evaluation.
    :return: Tuple of sets of scores for the Predicate, Predicate Sense, and Argument.
    """

    gold_file_conll_filename = ".".join(gold_file.split(".")[:-1]) + ".head.gold"
    gold_data = read_file(gold_file)
    gold_sen = get_conllu_sen(gold_data)

    pred_file_conll_filename = ".".join(pred_file.split(".")[:-1]) + ".head.pred"
    pred_data = read_file(pred_file)
    pred_sen = get_conllu_sen(pred_data)

    if conllu2x:
        gold_sen, pred_sen, pred_stat, argument_count = check_predicate_pos(gold_sen[0], pred_sen[0])

    if conllu2x:
        # Convert from CoNLL-U to CoNLL2009 format.
        logger.debug("Supposing both %s and %s files are in CoNLL-U format.", gold_file, pred_file)
        logger.debug("Converting %s from CoNLL-U format to CoNLL2009 format.", gold_file)
        gold_filename = conllu_conversion(gold_sen, gold_file_conll_filename, conversion='09')
        logger.debug("Converting %s from CoNLL-U format to CoNLL2009 format ", pred_file)
        pred_filename = conllu_conversion(pred_sen, pred_file_conll_filename, conversion='09')
        logger.debug("*****Evaluated only on common predicates in both the files*****")

        arg_stat, sense_stat = official_conll09_evaluation_script(gold_filename, pred_filename, argument_count, conditional_eval=conditional_eval)
    else:
        arg_stat, sense_stat = official_conll09_evaluation_script(gold_file, pred_file, argument_count, conditional_eval=conditional_eval)

    return pred_stat, sense_stat, arg_stat
