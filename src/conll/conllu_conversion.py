import collections
import codecs
import logging
from src.utils import conllu_format_check

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def convert(tup, di):
    """Convert a tuple into a dictionary.

    :param tup: List of tuples [(a, b), ...]
    :param di: Dictionary
    :return: Dictionary with tuple values as key-value pairs {a: b}
    """
    for a, b in tup:
        di.setdefault(a, b)
    return di


def read_file(filename, label=False):
    """Read the file, supports list of files.

    :param filename: Path to the file or a list of files.
    :param label: ??
    :return: Content of the file(s).
    """
    if isinstance(filename, str):
        with codecs.open(filename, 'r', 'utf-8') as f:
            data = f.readlines()
        return data
    elif isinstance(filename, list):
        data = []
        for file in filename:
            with codecs.open(file, 'r', 'utf-8') as f:
                data.extend(f.readlines())
            if label:
                data.append("OTHER")
            data.append("\n")
        return data
    else:
        logger.error("Filename format not recognized: %s", filename)


def get_conllu_sen(data):
    """Read in the CoNLL-U blocks.

    :param data:
    :return:
    """
    all_sen = []
    sen = []
    sen_id = []
    format_check = conllu_format_check(data)
    n_sen = 0
    if format_check < 0:
        return

    for line in data:
        tok = line.split("\t")
        if ((line == "\n") or (line == "\r\n")) and sen != []:
            all_sen.append(sen)
            sen = []
            n_sen = n_sen + 1
        elif tok[0][0] == "#":
            if tok[0].split(" ")[1] == "sent_id":
                sen_id.append(int(tok[0].split(" ")[-1].strip()))
            else:
                sen_id.append(tok[0].split("#")[1].strip())
            continue
        else:
            sen.append(line)
    if not sen_id:
        sen_id = [i for i in range(n_sen)]

    return all_sen, sen_id


def convert_bio(labels):
    """Convert into BIO format into argument span format.

    :param labels: List of labels in BIO format.
    :return: Converted format.
    """
    n = len(labels)
    tags = []

    tag = []
    count = 0

    # B I*
    for label in labels:
        count += 1

        if count == n:
            next_l = None
        else:
            next_l = labels[count]

        if label == "O":
            if tag:
                tags.append(tag)
                tag = []
            tags.append([label])
            continue

        tag.append(label[2:])

        if not next_l or next_l[0] == "B":
            tags.append(tag)
            tag = []

    new_tag = []

    for tag in tags:
        if len(tag) == 1:
            if tag[0] == "O":
                new_tag.append("*")
            else:
                new_tag.append("(" + tag[0] + "*)")
            continue

        label = tag[0]
        n = len(tag)

        for i in range(n):
            if i == 0:
                new_tag.append("(" + label + "*")
            elif i == n - 1:
                new_tag.append("*)")
            else:
                new_tag.append("*")

    return new_tag


def write_conll2009(data, filename):
    """Write the converted file in CoNLL2009 format.

    :param data: CoNLL2009 data.
    :param filename: Output file name.
    :return: None
    """
    f = open(filename, "w")
    for sen in data:
        for tok_line in sen:
            f.write("\t".join(tok_line))
        f.write("\n")
    f.close()
    logger.debug("Writing CoNLL2009 format done: %s", filename)


def write_conll2005(data, filename):
    """Write the converted file in CoNLL2005 format.

    :param data: CoNLL2005 data.
    :param filename: Output file name.
    :return: None
    """
    fout = open(filename, "w")
    for all_args in data:
        tokens = all_args[0]
        span_args = []
        for args in all_args[1:]:
            span_args.append(convert_bio(args))

        for label_column in span_args:
            assert len(label_column) == len(tokens)
        for i in range(len(tokens)):
            fout.write(tokens[i].ljust(15))
            for label_column in span_args:
                fout.write(label_column[i].rjust(15))
            fout.write("\n")
        fout.write("\n")
    fout.close()


def get_span_len(x):
    """Get the length of the span.

    :param x: List of tokens.
    :return:
    """
    ind = 0
    for tok in x:
        if tok != "O":
            ind += 1
    return ind


class SPAN:
    """Span class to store the span and the index info.
    """
    def __init__(self, span):
        self.span = span
        self.index = self.get_index()

    def get_index(self):
        index = []
        for ii, tok in enumerate(self.span):
            if tok != "O":
                index.append(ii)
        return index


def remove_overlapping_span(spans):
    """Remove the overlapping argument spans in the list of argument spans.

    :param spans: List of argument spans.
    :return: List of argument spans with overlaps removed.
    """
    if len(spans) == 1:
        return spans[0]
    else:
        spans.sort(key=get_span_len)
        final_args = spans[0]
        final_spans = SPAN(final_args)
        for span in spans[1:]:
            span = SPAN(span)
            for ii in span.index:
                if ii not in final_spans.index:
                    final_args[ii] = span.span[ii]
            final_spans = SPAN(final_args)
    process_args = []
    arg_set = set()
    for ii, tok in enumerate(final_args):
        if tok == "O":
            process_args.append(tok)
        else:
            arg = "-".join(tok.split("-")[1:])
            if arg not in arg_set:
                arg_set.add(arg)
                process_args.append("B-" + arg)
            else:
                if arg == "-".join(process_args[-1].split("-")[1:]):
                    process_args.append("I-" + arg)
                else:
                    process_args.append("O")

    return process_args


def get_span(tree, head):
    """Get the span of the head.

    IMPORTANT: This function is deprecated. Do not use except for CoNLL2005 evaluation.

    :param tree: Parse tree.
    :param head: Head node get build the span.
    :return: Span of the head node.
    """
    to_examine = collections.deque([head])
    span = []
    while len(to_examine) != 0:
        hea = to_examine.popleft()
        span.append(hea)
        indx = [k for k, v in tree.items() if v == hea]
        to_examine.extend(indx)
    return span


def get_2005_span(id_head, sen, predicate):
    """Get the argument span of the predicates.

    IMPORTANT: Only use for CoNLL2005 evaluation.

    :param id_head: Token ID of the head.
    :param sen: List of lines from the CoNLL2005 file.
    :param predicate: Number of predicates.
    :return: List of arguments.
    """
    id_head_dict = convert(id_head, {})

    #logger.debug("Number of predicates: %d", predicate)
    all_args = []
    for pred in range(predicate):
        all_arg_spans = []
        arg_ind = 10 + pred
        for line in sen:
            tok = line.strip().split("\t")
            arg_span = ["O" for _ in range(len(sen))]
            if tok[arg_ind] != "_":
                label = tok[arg_ind]
                target_span = get_span(id_head_dict, int(tok[0]))
                target_span.sort()
                #logger.debug("arg %s, target span %s", label, target_span)
                for ii, arg_index in enumerate(target_span):
                    if ii == 0:
                        arg_span[arg_index - 1] = "B-" + label
                    else:
                        arg_span[arg_index - 1] = "I-" + label
            all_arg_spans.append(arg_span)
        try:
            args = remove_overlapping_span(all_arg_spans)
            #logger.debug("Final argument list: %s", args)
            all_args.append(args)
        except:
            #logger.debug("Error in all_arg_spans: %s", all_arg_spans)
            pass

    return all_args


def conllu2conll2005(data):
    """Convert data from CoNLL-U format to CoNLL2005 format.

    :param data: CoNLL-U formatted data to convert.
    :return: Data in CoNLL2005 format.
    """
    all_pairs = []
    pairs = []
    txt = []
    tokens = []
    for sen in data:
        for line in sen:
            tok = line.strip().split("\t")
            pairs.append((int(tok[0]), int(tok[6])))
            tokens.append(tok[1])
        all_pairs.append(pairs)
        txt.append(" ".join(tokens))
        pairs = []
        tokens = []
    sen_span_args = []
    for ii, sent in enumerate(data):
        pred_tok = []
        n_predicates = len(data[ii][0].strip().split("\t")) - 10
        all_args = get_2005_span(all_pairs[ii], data[ii], n_predicates)
        all_args.insert(0, pred_tok)
        ind = 1
        for jj, tok_line in enumerate(sent):
            tok = tok_line.strip().split("\t")[9].split(".")
            if len(tok) > 1:
                # all_args[ind][jj] = "B-V"
                ind = ind + 1
                pred_tok.append(tok[0])
            else:
                pred_tok.append("-")
        sen_span_args.append(all_args)
    return sen_span_args


def conllu2conll2009(data):
    """Convert data from CoNLL-U format to CoNLL2009 format.

    :param data: CoNLL-U formatted data to convert.
    :return: Data in CoNLL2009 format.
    """
    data_09 = []
    for sen in data:
        sen_09 = []
        for tok_line in sen:
            tok = tok_line.split("\t")
            tok_09 = []
            kk = 0
            for element in range(13):
                if element == 3:
                    kk += 1
                    tok_09.append(tok[element - kk])
                elif element == 7:
                    kk += 1
                    tok_09.append(tok[element - kk])
                elif element == 9:
                    kk += 1
                    tok_09.append(tok[element - kk])
                elif element == 11:
                    kk += 1
                    tok_09.append(tok[element - kk])
                else:
                    tok_09.append(tok[element - kk])
            tok_09[6] = "_"
            tok_09[7] = "_"
            tok_09.extend(tok[9:])
            sen_09.append(tok_09)
        data_09.append(sen_09)
    return data_09


def conllu_conversion(conllu_sen, target_filename, conversion):
    """Convert CoNLL-U formatted data into CoNLL2005 or CoNLL2009 format.

    :param conllu_sen: CoNLL-U formatted file.
    :param target_filename: Output filename.
    :param conversion: Conversion format.
    :return: None
    """
    if conversion in ["2009", "09", "conll2009"]:
        conll09_filename = target_filename + ".conll09"
        data_09 = conllu2conll2009(conllu_sen)
        write_conll2009(data_09, conll09_filename)
        return conll09_filename
    elif conversion in ["2005", "05", "conll2005"]:
        conll05_filename = target_filename + ".conll05"
        data_05 = conllu2conll2005(conllu_sen)
        write_conll2005(data_05, conll05_filename)
        return conll05_filename
    else:
        logger.error("Conversion format not recognized: %s", conversion)
