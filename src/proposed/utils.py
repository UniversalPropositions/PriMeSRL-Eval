from collections import OrderedDict
import re
import logging
from src.utils import conllu_format_check

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def output_df(df, fn):
    """Wrapper to write the dataframe to the given file name.

    :param df: Dataframe to output.
    :param fn: Output filename.
    :return: None
    """
    df.to_csv(fn, index=False)
    logger.info('Saved: %s', fn)


def get_conllu_column_names():
    """Get the CoNLL-U format column names. UD format: https://universaldependencies.org/format.html
    Format extended with "FLAG" (whether a token is a predicate) and "VERB" (predicate sense).

    :return: Column names.
    """
    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "FLAG", "VERB"]

    return columns


class MetaInfo:
    """Class for the metadata info of the CoNLL-U block.
    """
    def __init__(self, meta):
        self.meta = meta
        self.sen_id = ""
        self.sen_txt = ""
        self.process_meta()

    def process_meta(self):
        """Process the meta info of the CoNLL-U block.
        """
        for meta_line in self.meta:
            if re.match(r'^# [0-9]+ #', meta_line):  # len(meta_line.split("#")) >= 3: #
                self.sen_id = int(meta_line.split("#")[1])
                self.sen_txt = "#".join(meta_line.split("#")[2:]).strip()
            elif len(meta_line.split("sentence-text:")) >= 2:
                self.sen_txt = "sentence-text:".join(meta_line.split("sentence-text:")[1:]).strip()
            elif len(meta_line.split("text = ")) >= 2:
                self.sen_txt = "text = ".join(meta_line.split("text = ")[1:]).strip()
            elif len(meta_line.split("sentence-text:")) == 1 and not re.match(r'^# \d # ', meta_line):
                self.sen_txt = "#".join(meta_line.split("#")[1:]).strip()


class Token:
    """Class for the token and the column values in the CoNLL-U block.
    """
    def __init__(self, tok_line, col_names, sen):
        self.id = int(tok_line[col_names.index("ID")])
        self.form = tok_line[col_names.index("FORM")]
        self.lemma = tok_line[col_names.index("LEMMA")]
        self.upos = tok_line[col_names.index("UPOS")]
        if "XPOS" in col_names:
            self.xpos = tok_line[col_names.index("XPOS")]
        else:
            self.xpos = "_"
        self.feat = tok_line[col_names.index("FEATS")]
        self.head = int(tok_line[col_names.index("HEAD")])
        self.deprel = tok_line[col_names.index("DEPREL")]
        self.ispred = self.is_pred(tok_line[col_names.index("FLAG")])
        self.sense = self.sense(tok_line[col_names.index("VERB")])
        self.children = self.get_child(col_names, sen)

    def is_pred(self, tok):
        """Check if the token is a predicate.

        :param tok: Token to check.
        :return: True if the token is not a "_" value.
        """
        if tok == "_":
            return False
        else:
            return True

    def sense(self, tok):
        """Get the predicate sense of the token, form "buy.01".

        :param tok: Token to get the sense.
        :return: Sense number of the token.
        """
        if not self.is_pred(tok):
            return "_"
        else:
            return tok.split(".")[-1]

    def get_child(self, col_names, sen):
        """Get the child of this token.

        :param col_names: Column names of the columns in the CoNLL-U block.
        :param sen: CoNLL-U block of the sentence.
        :return: The token IDs of the children of this token.
        """
        child = []
        deprel_index = col_names.index("HEAD")
        for tok_id, tok_line in enumerate(sen):
            if int(tok_line[deprel_index]) == self.id:
                child.append(tok_id + 1)
        return child

    def __str__(self):
        """Default string format for printing.

        :return: Formatted string of the token and its attributes.
        """
        out = "id: {}, form: {}, lemma: {}, upos: {}, xpos: {}, feat: {}, head: {}, deprel: {}, pred: {}".format(
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feat,
            self.head,
            self.deprel,
            self.ispred
        )
        return out


class SenChunk:
    """Class for one CoNLL-U block.
    """
    def __init__(self, sen, sen_count, col_names=[]):
        self.sen, self.meta = self.get_sen_meta(sen)
        self.if_no_predicate = 0
        self.n_predicate = 0
        self.n_columns = len(sen[0])
        if not col_names:
            self.col_names = get_conllu_column_names()
        else:
            self.col_names = col_names
        self.predicates = self.get_predicate()
        self.arguments = self.get_arguments()
        self.txt = self.get_text()
        self.n_token = len(self.txt.split(" "))
        self.deprel = self.get_deprel()
        self.tokens = self.get_tokens()
        self.sen_id = self.get_sen_id(sen_count)

    def get_sen_meta(self, sen):
        """Separate the sentence from meta information,

        :param sen: List of tab separated text
        :return: Sentence text and metadata info.
        """
        sent = []
        meta = []
        for tok_line in sen:
            if tok_line[0].split(" ")[0] == "#":
                meta.append(" ".join(tok_line))
            else:
                sent.append(tok_line)

        meta_out = MetaInfo(meta)
        return sent, meta_out

    def get_predicate(self):
        """Get the predicates in a CoNLL-U block.

        :return: List of predicates in a sentence, each predicate is a tuple (verb_id, verb, verb.sense, pos)
        """
        if len(self.sen[0]) <= len(self.col_names):
            self.if_no_predicate += 1
            return []
        predicates = []
        for tok_line in self.sen:
            if tok_line[self.col_names.index("FLAG")] == "Y":
                predicates.append((tok_line[0],
                                   tok_line[self.col_names.index("VERB")].split(".")[0],
                                   tok_line[self.col_names.index("VERB")],
                                   tok_line[self.col_names.index("UPOS")]))

                self.n_predicate += 1
        return predicates

    def get_arguments(self):
        """Get the arguments in the CoNLL-U block.

        :return: List of argument columns.
        """
        arg_columns = list(map(list, zip(*self.sen)))[len(self.col_names):]
        return arg_columns

    def get_text(self):
        """Get the text of the sentence in the metadata.

        :return: Sentence text.
        """
        text = " ".join(list(zip(*self.sen))[1])
        if self.meta.sen_txt != "":
            return self.meta.sen_txt
        return text

    def get_deprel(self):
        """Get the DEPREL column.

        :return: Value of the DEPREL column.
        """
        deprel = []
        for tok_line in self.sen:
            deprel.append(tok_line[self.col_names.index("DEPREL")])
        return deprel

    def get_tokens(self):
        """Get the tokens.

        :return: List of the tokens.
        """
        tokens = []
        for tok_line in self.sen:
            tok = Token(tok_line, self.col_names, self.sen)
            tokens.append(tok)
        return tokens

    def get_sen_id(self, sen_count):
        """Get the sentence ID. If CoNLL-U block has no sentence ID, use the
        numbered occurrence of the block in the file.

        :param sen_count: Numbered position of the CoNLL-U block in the file.
        :return: Sentence ID.
        """
        if self.meta.sen_id == "":
            return sen_count
        else:
            return int(self.meta.sen_id)


class ReadData:
    """Class to read the CoNLL-U data into an internal representation.
    """
    def __init__(self, filename):
        self.filename = filename
        self.sen_with_no_predicates = 0
        self.col_names = self.get_column_names()
        self.all_sen = self.get_sen()
        self.n_sen = len(self.all_sen)
        self.predicate_count = self.count_predicates()

    def read_file(self):
        """Read the CoNLL-U formatted file.

        :return:
        """
        with open(self.filename, 'r') as f:
            data = f.readlines()
        check_new_lines = conllu_format_check(data)
        if check_new_lines == -1:
            logger.error("Bad CoNLL-U format: %s", self.filename)
            return None
        else:
            return data

    def get_sen(self):
        """Parse the CoNLL-U blocks.

        :return: All CoNLL-U blocks as SenChunk class.
        """
        all_sen = []
        sen = []
        tok_lines = self.read_file()
        cont_sen = 0
        for tok_line in tok_lines:
            if tok_line == "\n" or tok_line == "---\n" or tok_line == "\r\n":
                if sen:
                    chunk = SenChunk(sen, cont_sen, self.col_names)
                    all_sen.append(chunk)
                    self.sen_with_no_predicates += chunk.if_no_predicate
                sen = []
            else:
                sen.append(tok_line.strip().split("\t"))

        if sen:
            all_sen.append(SenChunk(sen, self.col_names))
            self.sen_with_no_predicates += SenChunk(sen, self.col_names).if_no_predicate
        return all_sen

    def count_predicates(self):
        """Count all the predicates in all CoNLL-U blocks.

        :return: Count.
        """
        count = 0
        for sen in self.all_sen:
            count += len(sen.predicates)
        return count

    def get_predicate_stat(self):
        """Count the predicates.

        :return: Dictionary of the predicate counts.
        """
        stat = OrderedDict()
        for sen in self.all_sen:
            if sen.n_predicate > 0:
                for pred in sen.predicates:
                    if pred[-1] in stat:

                        stat[pred[-1]] += 1
                    else:
                        stat[pred[-1]] = 1
        return stat

    def get_arg_stat(self):
        """Count the arguments.

        :return: Dictionary of the argument counts.
        """
        arg_count = {}
        for sen in self.all_sen:
            for arg in sen.arguments:
                for tok_arg in arg:
                    if tok_arg != "_":
                        if tok_arg not in arg_count:
                            arg_count[tok_arg] = 1
                        else:
                            arg_count[tok_arg] += 1
        return arg_count

    def get_column_names(self):
        return []


class Reader(ReadData):
    """Reader class to parse the CoNLL-U file.
    """
    def __init__(self, input_file, data_format):
        self.data_format = data_format
        super(Reader, self).__init__(input_file)

    def get_column_names(self):
        if self.data_format == "conllu":
            return get_conllu_column_names()
        else:
            logger.error("No data format specified.")
