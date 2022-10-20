import argparse
import logging
from src.pipeline import proposed_evaluation, official_conll_evaluation
from src.utils import format_dataframe
from src.reader.reader import Reader

LOGGING_FORMAT = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

################################################################################
################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate two given CoNNL-U+SRL files.')
    parser.add_argument('-g', '--gold-conllu', type=str, default="data/CoNLL2009-ST-evaluation-English-ood.txt", help='File path of CoNLL-U with gold SRL.')
    parser.add_argument('-p', '--pred-conllu', type=str, default="data/srl.head.pred.conll09", help='File path of CoNNL-U with predicted SRL.')
    parser.add_argument('-o', '--output-folder', type=str, default="tmp", help='Output folder for files.')
    parser.add_argument('-f', '--format', type=str, default="conllu", help='[conll09, conll05, conllu] default=conllu')
    args = parser.parse_args()

    gold_conllu_fp = args.gold_conllu
    pred_conllu_fp = args.pred_conllu
    output_folder = args.output_folder
    file_format = args.format = "conll09"

    cols = ['Metric', 'Type', 'Precision', 'Recall', 'F1']

    ## Format converter
    # Internal format accepted in conllu
    # All other formats are first converted into conllu format before processing

    if file_format in ["2009", "09", "conll2009", "conll09"]:
        data_09 = Reader(gold_conllu_fp, "conll09")
        gold_conllu_fp = gold_conllu_fp + ".conllu"
        data_09.write_conllu(gold_conllu_fp)
        data_09 = Reader(pred_conllu_fp, "conll09")
        pred_conllu_fp = pred_conllu_fp + ".conllu"
        data_09.write_conllu(pred_conllu_fp)
        # write_conll2009(data_09, conll09_filename)
    elif file_format in ["2005", "05", "conll2005", "conll05"]:
        gold_conllu_fp = gold_conllu_fp + ".conllu"
        pred_conllu_fp = pred_conllu_fp + ".conllu"
    elif file_format in ["conllu"]:
        gold_conllu_fp = gold_conllu_fp
        pred_conllu_fp = pred_conllu_fp
    else:
        logger.error("Conversion format not recognized: %s", file_format)

    df_c = official_conll_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)
    df_p = proposed_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)

    print("*" * 80)
    print("Official CoNLL (2005 and 2009) evaluation:")
    print(format_dataframe(df_c[cols]))
    print("*" * 80)

    print("*" * 80)
    print("Proposed evaluation:")
    print(format_dataframe(df_p[cols]))
    print("*" * 80)
