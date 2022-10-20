# CoNLLU Toolkit: Description of column names for different formats
#
# Author: Ishan Jindal <ishan.jindal@ibm.com>
#

import codecs

def get_conllu_column_names():
    # There are the columns name taken from UD data
    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "FLAG", "VERB"]

    # ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
    # FORM: Word form or punctuation symbol.
    # LEMMA: Lemma or stem of word form.
    # UPOS: Universal part-of-speech tag.
    # XPOS: Language-specific part-of-speech tag; underscore if not available.
    # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    # HEAD: Head of the current word, which is either a value of ID or zero (0).
    # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    # 4	doivent	devoir	VERB	VERB	_	0	root	_	_
    # 14	ait	avoir	VERB	VERB	_	10	ccomp	_	_

    return columns


def get_conll09_column_names():
    # These column names are from the CoNLL2009 propbank data.
    # Changed POS to UPOS to have consistency among different formats.
    columns = ["ID", "FORM", "LEMMA","PLEMMA" ,"UPOS", "PUPOS", "FEATS","PFEATS", "HEAD", "PHEAD","DEPREL", "PDEPREL","FLAG", "VERB"]
    return columns


def get_conllus_column_names():
    # CoNLLus does not contains column for predicate flag
    # It include the predicate information in the argument columns
    columns = ["ID", "FORM", "LEMMA","PLEMMA" ,"UPOS", "PUPOS", "FEATS","PFEATS", "HEAD", "PHEAD","DEPREL", "PDEPREL", "VERB"]
    return columns


class FormatChecker():

    def __init__(self, filename, format):
        self.filename = filename
        self.format = format
        self.data = self.read_file()
        self.error_count = 0
        self.check_n_lines()
        self.check_format()

    def read_file(self):
        with codecs.open(self.filename, 'r', 'utf-8') as f:
            data = f.readlines()
        return data

    def check_n_lines(self):
        """
        Expecting two empty lines at the end of the file
        :return:
        """

        # just checking new lines at the end of conll u file
        end_line = -1
        count_new_lines = 0
        while (self.data[end_line] == "\n") or (self.data[end_line] == "\r\n"):
            count_new_lines += 1
            end_line = end_line -1
        if count_new_lines != 1:
            print("Expecting a new line at the end of the file.")
            print("There are {} new lines".format(count_new_lines))
            self.error_count += 1
        else:
            print("Numbers of lines at the end of the file is UD compliant")

    def check_format(self):
        if self.format == "conllu":
            check_conllu_format(self.data)
        elif self.format == "conllus":
            check_conllus_format(self.data)
        elif self.format == "conllu_json":
            check_conllu_json_format(self.data)
        elif self.format == "conll09":
            check_conll09_format(self.data)

def check_conllu_format(data):

    return


def check_conllus_format(data):
    return

def check_conllu_json_format(data):
    return

def check_conll09_format(data):
    return
