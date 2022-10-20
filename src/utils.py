import os
import shutil
import requests
import hashlib
import logging
import pandas as pd
import codecs

from src.reader.reader import SenChunk


logger = logging.getLogger(__name__)

conll05_eval_script_path = os.path.join(*['src', 'conll', 'conll2005', 'srl-eval.pl'])
conll05_eval_script_url = "https://www.cs.upc.edu/%7Esrlconll/srl-eval.pl"
conll05_eval_script_sha1 = "5be33e0aae10c83e4d0bef20d322a7048d0d2907"

conll09_eval_script_path = os.path.join(*['src', 'conll', 'conll2009', 'eval09.pl'])
conll09_eval_script_url = "https://ufal.mff.cuni.cz/conll2009-st/eval09.pl"
conll09_eval_script_sha1 = "1f245a1812b08651d74e5c73b8856f348da76c53"

cols_int = {'Correct', 'Missing', 'Spurious', 'Incorrect', 'SameValueSameSpanDiffHead', 'SameSpanDiffValue',
            'SameHeadSameValueDiffSpan', 'SameValueDiffSpan'}
cols_float = {'Precision', 'Recall', 'F1'}

################################################################################
################################################################################


def sha1_hash(fp):
    with open(fp, 'r') as f:
        return hashlib.sha1(f.read().encode('utf-8')).hexdigest()


def download_file(url, fp):
    with open(fp, 'wb') as f:
        content = requests.get(url).content
        f.write(content)


def download_official_scripts():
    """Download the official CoNLL evaluation scripts and place them in the appropriate local folder.
    """

    if os.path.isfile(conll05_eval_script_path):
        logger.info("Local CoNLL 2005 official script path: %s", conll05_eval_script_path)
        local_conll05_sha1 = sha1_hash(conll05_eval_script_path)
        if local_conll05_sha1 != conll05_eval_script_sha1:
            logger.warning("Local CoNLL 2005 official script differs from expected script (SHA-1 hash differs). Expected: %s, local: %s", conll05_eval_script_sha1, local_conll05_sha1)
    else:
        logger.info("Downloading CoNLL 2005 official evaluation script from: %s", conll05_eval_script_url)
        download_file(conll05_eval_script_url, conll05_eval_script_path)
        local_conll05_sha1 = sha1_hash(conll05_eval_script_path)
        logger.info("Local CoNLL 2005 official script path %s:", conll05_eval_script_path)
        if local_conll05_sha1 != conll05_eval_script_sha1:
            logger.warning("Local CoNLL 2005 official script differs from downloaded script (SHA-1 hash differs). Expected: %s, local: %s", conll05_eval_script_sha1, local_conll05_sha1)

    if os.path.isfile(conll09_eval_script_path):
        logger.info("Local CoNLL 2005 official script path: %s", conll09_eval_script_path)
        local_conll09_sha1 = sha1_hash(conll09_eval_script_path)
        if local_conll09_sha1 != conll09_eval_script_sha1:
            logger.warning("Local CoNLL 2009 official script differs from expected script (SHA-1 hash differs). Expected: %s, local: %s", conll09_eval_script_sha1, local_conll09_sha1)
    else:
        logger.info("Downloading CoNLL 2009 official evaluation script from: %s", conll09_eval_script_url)
        download_file(conll09_eval_script_url, conll09_eval_script_path)
        local_conll09_sha1 = sha1_hash(conll09_eval_script_path)
        logger.info("Local CoNLL 2005 official script path: %s", conll09_eval_script_path)
        if local_conll09_sha1 != conll09_eval_script_sha1:
            logger.warning("Local CoNLL 2009 official script differs from expected script (SHA-1 hash differs). Expected: %s, local: %s", conll09_eval_script_sha1, local_conll09_sha1)


def format_dataframe(df_):
    """Format the dataframe for human viewing.

    :param df_: Dataframe with raw quality numbers.
    :return: Formatted dataframe.
    """
    if df_ is None:
        return None

    df = df_.copy()
    for col in df.columns:
        if col in cols_int:
            df[col] = df[col].astype(int)
        elif col in cols_float:
            df[col] = df[col].apply(lambda x: "{0:.2f}%".format(x*100))
    return df


def setup_output_folders(gold_conllu_fp, pred_conllu_fp, output_folder):
    """Setup the output folders to store the intermediate files.

    :param gold_conllu_fp: Path to the gold file in CoNLL-U format.
    :param pred_conllu_fp: Path to the pred file in CoNLL-U format.
    :param output_folder: Path to the output folder.
    :return: Basename of each file, parent folder of each file, path to each file, folder to store the comparisons
    """
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Setting up folders for output in: %s", output_folder)

    gold_basename = os.path.splitext(os.path.basename(gold_conllu_fp))[0]
    pred_basename = os.path.splitext(os.path.basename(pred_conllu_fp))[0]

    gold_path = os.path.join(output_folder, gold_basename)
    pred_path = os.path.join(output_folder, pred_basename)

    os.makedirs(gold_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    out_gold_file = os.path.join(gold_path, "srl.conllu")
    out_pred_file = os.path.join(pred_path, "srl.conllu")
    shutil.copy(gold_conllu_fp, out_gold_file)
    shutil.copy(pred_conllu_fp, out_pred_file)

    compare_output_folder = os.path.join(output_folder, 'compare-' + gold_basename + '-' + pred_basename)
    os.makedirs(compare_output_folder, exist_ok=True)

    return gold_basename, pred_basename, gold_path, pred_path, out_gold_file, out_pred_file, compare_output_folder


def conllu_format_check(data):
    """Check the format of the CoNLL-U file to ensure it has the correct number
    of newlines at the end of the file.

    :param data: Parsed CoNLL-U data.
    :return: Number of new lines at the end of the CoNLL-U file.
    """
    end_line = -1
    count_new_lines = 0
    while (data[end_line] == "\n") or (data[end_line] == "\r\n"):
        count_new_lines += 1
        end_line = end_line - 1
    if count_new_lines != 1:
        logger.error("Expecting a new line at the end of the file.")
        logger.error("There are %d new lines.", count_new_lines)
        return -1
    else:
        return 0


def write_raw_csv(conllu_file):
    """Write the sentences from the conllu file.
    """
    path_raw_sen = os.path.dirname(conllu_file)+"/"+"sentences.csv"
    # path_raw_sen = os.path.join(conllu_file.split("/")[:-1])+".csv"
    print(path_raw_sen)
    sen_id =[]
    text = []
    ind = 0
    for sen in read_large_data(conllu_file):
        sen_id.append(ind)
        sen = SenChunk(sen)
        ind += 1
        text.append(sen.txt)
    df = pd.DataFrame(zip(sen_id, text), columns=["id", "text"])
    df.to_csv(path_raw_sen,index=False)
    return path_raw_sen


def read_large_data(filename):
    """Reads one sentence at a time from a large CoNLL file.
    """
    with codecs.open(filename, 'r', 'utf-8') as f:
        data = f.readlines()

    all_sen = []
    sen = []
    for line in data:
        if line == "\n" or line == "---\n" or line == "\r\n":
            if sen != []:
                yield sen
            sen = []
        else:
            sen.append(line)
    if sen != []:
        yield sen


