# Reader for CoNLL formats
#
# Author: Ishan Jindal <ishan.jindal@ibm.com>
#

from collections import OrderedDict
from src.reader.meta_reader import MetaInfo
from src.format_description import *


def get_conllu_sen_from_conllus(conllus_sen):
    """Extract the conllu format components from the conllus format.

    :param conllus_sen: Sentence blocks in conllus format
    :type conllus_sen: list of [str]

    :return: Sentence blocks in ESSP conllu format
    """
    conllu_sen = []
    for sen in conllus_sen:
        sen_len = len(sen)
        us_sen = list(map(list, zip(*sen)))
        del us_sen[8:10]
        pred_flag = ["_" for _ in range(sen_len)]
        ind = 0
        for ii in range(sen_len):
            # Format seems to have changed slightly and is missing a column for some blocks.
            # Need to insert a new column.
            if len(us_sen) == 8:
                us_sen.append(['_' for _ in range(len(us_sen[0]))])
            if us_sen[8][ii] != "_":
                ind = ind + 1
                pred_flag[ii] = "Y"
                us_sen[8 + ind][ii] = "_"
        us_sen.insert(8, pred_flag)
        for col_id, arg_col in enumerate(us_sen[10:]):
            for tok_id, arg in enumerate(arg_col):
                us_sen[10+col_id][tok_id] = arg.replace("ARG", "A")
        conllu_sen.append(list(map(list, zip(*us_sen))))
    return conllu_sen

def get_conllu_sen_from_conll09(sen):
    conllu_sen = []
    conll09_col = get_conll09_column_names()
    conllu_col = get_conllu_column_names()
    sen = list(map(list, zip(*sen)))
    srl_ind = conll09_col.index("VERB") + 1
    srl = sen[srl_ind:]
    for n_cl in conllu_col:
        if n_cl in conll09_col:
            conllu_sen.append(sen[conll09_col.index(n_cl)])
        else:
            conllu_sen.append(["_" for _ in sen[0]])
    conllu_sen.extend(srl)
    conllu_sen = list(map(list, zip(*conllu_sen)))
    return conllu_sen

class Token():
    """Represents the attributes of each token.
    """
    def __init__(self, tok_line, col_names, sen):
        self.id = tok_line[col_names.index("ID")]
        self.form = tok_line[col_names.index("FORM")]
        self.lemma = tok_line[col_names.index("LEMMA")]
        self.upos = tok_line[col_names.index("UPOS")]
        if "XPOS" in col_names:
            self.xpos = tok_line[col_names.index("XPOS")]
        else:
            self.xpos = "_"
        self.feat = tok_line[col_names.index("FEATS")]
        self.head = tok_line[col_names.index("HEAD")]
        self.deprel = tok_line[col_names.index("DEPREL")]
        self.ispred = self.is_pred(tok_line[col_names.index("VERB")])
        self.sense = self.sense(tok_line[col_names.index("VERB")])
        self.children = self.get_child(col_names, sen)
        self.start = None
        self.end = None

    def is_pred(self, tok):
        if tok == "_":
            return False
        else:
            return True

    def sense(self, tok):
        if not self.is_pred(tok):
            return "_"
        else:
            return tok.split(".")[-1]

    def get_child(self, col_names, sen):
        child = []
        deprel_index = col_names.index("HEAD")
        for tok_id, tok_line in enumerate(sen):
            if tok_line[deprel_index] == self.id:
                child.append(tok_id+1)
        return child


    def __str__(self):
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


class SenChunk():
    """Represents each sentence block/chunk.
    """
    def __init__(self, sen, sen_count, col_names=[], data_format="conllu"):
        self.sen, self.meta = self.get_sen_meta(sen, data_format)
        # if self.meta.srl != "":
        #     sen = list(map(list, zip(*self.sen)))[:-2]
        #     srl_block = self.get_srl_block(self.meta.srl, len(sen[0]))
        #     sen.extend(srl_block)
        #     sen = list(map(list, zip(*sen)))
        #     self.sen = sen
        self.if_no_predicate = 0
        self.n_predicate = 0
        self.n_columns = len(sen[0])
        if col_names == [] or data_format in ["conllus", "conll09"] :
            # print("Assuming CoNLLu formatted columns")
            self.col_names = get_conllu_column_names()
        else:
            self.col_names = col_names
        self.predicates = self.get_predicate(col_names)
        self.arguments = self.get_arguments()
        self.txt = self.get_text() #self.meta.sen_txt
        self.n_token = len(self.txt.split(" "))
        self.deprel = self.get_deprel()
        self.tokens = self.get_tokens()
        self.sen_id = self.get_sen_id(sen_count)

    def get_sen_meta(self, sen, data_format):
        """
        Seperate out the sentence from meta information
        :param sen: List of tab seperated text
        :return:
        """
        sent = []
        meta = []
        for tok_line in sen:
            if tok_line[0].split(" ")[0] == "#":
                meta.append(" ".join(tok_line))
            # elif len(tok_line[0].split("-")) > 1: # Do not read multiword tokens
            #     continue
            # elif len(tok_line[0].split(".")) > 1:  # Do not read multi-token words
            #     continue
            else:
                sent.append(tok_line)

        meta_out = MetaInfo(meta)

        if meta_out.srl != "":
            sen = list(map(list, zip(*sent)))[:-2]
            srl_block = self.get_srl_block(meta_out.srl, len(sen[0]))
            sen.extend(srl_block)
            sen = list(map(list, zip(*sen)))
            sent = sen

        if data_format == "conllus":
            sent = get_conllu_sen_from_conllus([sent])[0]

        if data_format == "conll09":
            sent = get_conllu_sen_from_conll09(sent)

        return sent, meta_out

    def get_predicate(self, col_names):
        """
        :return: a list of predicates in a sentence, where each pred is a tuple (verb_id, verb, verb.sense, pos)
        """
        if len(self.sen[0]) <= len(self.col_names):
            self.if_no_predicate += 1
            # print("no predicate")
            return []
        # ind = 0
        predicates = []
        for tok_line in self.sen:
            # tok_col = tok.strip().split("\t")
            # for feat in tok_col:
            if tok_line[self.col_names.index("VERB")] != "_":
                # start, end = get_tok_position(int(tok_col[0])-1, int(tok_col[0])-1, tokens)
                # get_tok_position(tok_col[0]-1, tokens)
                predicates.append((tok_line[0],
                                   tok_line[self.col_names.index("VERB")].split(".")[0],
                                   tok_line[self.col_names.index("VERB")],
                                   tok_line[self.col_names.index("UPOS")]))

                self.n_predicate += 1
        return predicates

    def get_arguments(self):
        args = OrderedDict()
        arg_columns = list(map(list,zip(*self.sen)))[len(self.col_names):]
        # for ii, pred in enumerate(self.predicates):
        #     args[pred] = arg_columns[ii]
        return arg_columns

    def get_text(self):
        text = " ".join(list(zip(*self.sen))[1])
        if self.meta.sen_txt != "":
            return self.meta.sen_txt
        return text

    def get_deprel(self):
        deprel = []
        for tok_line in self.sen:
            deprel.append(tok_line[self.col_names.index("DEPREL")])
        return deprel

    def get_tokens(self):
        tokens = []
        for tok_line in self.sen:
            tok = Token(tok_line, self.col_names, self.sen)
            # print(tok)
            tokens.append(tok)
        return tokens

    def get_sen_id(self,sen_count):
        if self.meta.sen_id == "":
            return str(sen_count)
        else:
            return str(self.meta.sen_id)

    @staticmethod
    def get_srl_block(srl_json, sen_len):
        """
        Convert Json formatted SRL block into CoNLLu SRL format
        :param srl_json: json object
        :param sen_len: Number of tokens in sentence
        :return:
        """

        n_predicates = len(srl_json["verbs"])
        srl_block = [["_" for _ in range(sen_len)] for _ in range(n_predicates + 2)]
        for pred_id, block in enumerate(srl_json["verbs"]):
            srl_block[0][block["verb"][0] - 1] = "Y"
            srl_block[1][block["verb"][0] - 1] = block["verb"][1]
            for args in block["arguments"]:
                if args[1].split("-")[-1] != "V": ## this transformation removes "v", "c-v", or "r-v" tags from the arguments
                    srl_block[2+pred_id][args[0]-1] = args[1].replace("ARG", "A")
        return srl_block

    # @staticmethod
    # def get_srl_block(srl_json, sen_len):
    #     """
    #     Convert Json formatted SRL block into CoNLLu SRL format
    #     :param srl_json: json object
    #     :param sen_len: Number of tokens in sentence
    #     :return:
    #     """
    #     n_predicates = len(srl_json)
    #     srl_block = [["_" for _ in range(sen_len)] for __ in range(n_predicates + 2)]
    #     for pred_id, block in enumerate(srl_json):
    #         srl_block[0][block["target"] - 1] = "Y"
    #         srl_block[1][block["target"] - 1] = block["sense"]
    #         for args in block["args"]:
    #             if args[1].split("-")[-1] != "V": ## this transformation removes "v", "c-v", or "r-v" tags from the arguments
    #                 srl_block[2+pred_id][args[0]-1] = args[1].replace("ARG", "A")
    #     return srl_block

class ReadData():

    def __init__(self, filename, sen_dict=""):
        self.filename = filename
        self.sen_with_no_predicates = 0
        self.col_names = self.get_column_names()
        self.all_sen = self.get_sen(sen_dict)
        self.n_sen = len(self.all_sen)
        # self.predicate_count = self.count_predicates()
        self.data_format = self.get_data_format()
        FormatChecker(self.filename, self.data_format)

    def read_file(self):
        #with open(self.filename) as f:
        with codecs.open(self.filename, 'r', 'utf-8') as f:
            data = f.readlines()
        # chek_new_lines = conllu_format_check(self.filename)
        # if chek_new_lines == -1:
        #     print("Fix number of lines at the end.")
        #     return None
        # else:
        #     # print("Checked number of new lines at the end of the document.")
        return data

    def get_sen(self,sen_dict):
        if sen_dict:
            all_sen = {}
        else:
            all_sen = []
        sen = []
        tok_lines = self.read_file()
        count_sen = 0
        for tok_line in tok_lines:
            if tok_line == "---\n" or tok_line.strip() == "":
                if sen != []:
                    chunk = SenChunk(sen, count_sen, self.col_names, self.data_format)
                    if sen_dict:
                        if chunk.sen_id in all_sen:
                                all_sen[chunk.sen_id].append(chunk)
                        else:
                            all_sen[chunk.sen_id] = [chunk]
                    else:
                        all_sen.append(chunk)
                    self.sen_with_no_predicates += chunk.if_no_predicate
                sen = []
                count_sen += 1
            # elif line[0] == "#":  # remove all meta data
            #     continue
            else:
                sen.append(tok_line.strip().split("\t"))

        if sen != []:
            chunk = SenChunk(sen, count_sen, self.col_names, self.data_format)
            if sen_dict:
                if chunk.sen_id in all_sen:
                    all_sen[chunk.sen_id].append(chunk)
                else:
                    all_sen[chunk.sen_id] = [chunk]
            else:
                all_sen.append(chunk)
                all_sen.append(chunk)
            self.sen_with_no_predicates += chunk.if_no_predicate
        return all_sen

    def count_predicates(self):
        count = 0
        for sen in self.all_sen:
            count += len(sen.predicates)
        return count

    def get_predicate_stat(self):
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
        arg_count = {}
        for sen in self.all_sen:
            for arg in sen.get_arguments():
                #print(arg)
                for tok_arg in arg:
                    if tok_arg != "_":
                        if tok_arg not in arg_count:
                            arg_count[tok_arg] = 1
                        else:
                            arg_count[tok_arg] += 1
        return arg_count


    def get_column_names(self):
        return []

    def get_data_format(self):
        print("No data format defined. Asssuming `CoNLLu` format")
        return "conllu"


class Reader(ReadData):
    """Reader that processes the CoNLL data.
    """

    def __init__(self, input_file, data_format, sen_dict=""):
        self.data_format = data_format
        super(Reader, self).__init__(input_file, sen_dict)

    def get_column_names(self):
        if self.data_format == "conllu":
            return get_conllu_column_names()
        elif self.data_format == "conll09":
            return get_conll09_column_names()
        elif self.data_format == "conllus":
            return get_conllu_column_names()
        else:
            print("No format specified, need: conllu or conll09.")

    def write_conllu(self, filename, data=[], keep_meta=False):

        f_out = open(filename, "w", encoding='utf-8')
        ''' Specify if want to write all sentence in self or a new set of sentences.'''
        if data == []:
            all_sen = self.all_sen
        else:
            all_sen = data

        for sen_id, sen in enumerate(all_sen):
            sent_id = sen_id
            if sen.meta.sen_id:
                sent_id = sen.meta.sen_id

            '''Specify if we want to keep entire meta data else print only sentence id and sentence text'''
            if keep_meta:
                for tok_line in sen.meta.meta:
                    f_out.write(tok_line+"\n")
            else:
                # f_out.write("# sent_id = {}\n".format(sent_id))
                f_out.write("# text = {}\n".format(sen.txt))
            for tok_line in sen.sen:
                f_out.write("\t".join(tok_line) + "\n")
            f_out.write("\n")
        f_out.close()
        print("\nOutput conllu file with SRL labels is written at {}".format(filename))
        return

    def write_raw_sen(self, fn="", type="raw"):
        if type == "tok":
            if fn == "":
                fn = ".".join(self.filename.split(".")[:-1])+".tok.txt"
            f = open(fn, "w")
            for sen in self.all_sen:
                for tok_line in sen.sen:
                    f.write(tok_line[1]+"\n")
                f.write("\n")
            f.close()
        elif type == "raw":
            if fn == "":
                fn = ".".join(self.filename.split(".")[:-1])+".raw.txt"
            f = open(fn, "w")
            for sen in self.all_sen:
                f.write(sen.txt+"\n")
            f.close()

    def write_senid2txt_map(self, fn=""):
        import csv
        if fn == "":
            fn = ".".join(self.filename.split(".")[:-1]) + ".sentid2txt.csv"
        fields = ["id", "text"]
        rows = []
        for sen in self.all_sen:
            rows.append([str(sen.meta.sen_id), sen.txt])

        # writing to csv file
        with open(fn, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(fields)

            # writing the data rows
            csvwriter.writerows(rows)
        return

    def get_data_format(self):
        return self.data_format


if __name__ == "__main__":
    fn = "/Users/ishan/git/ConlluToolKit/data/test/reader/conllu_reader/input/test1.conllu"
    data = Reader(fn, "conllu")
    print("DONE")
    data.write_conllu(fn+".conllu")
