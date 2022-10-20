import pandas as pd
import collections
import logging
from src.proposed.utils import Reader

logger = logging.getLogger(__name__)

VBZ_names = ["VERB", "PART", "AUX", "VB", "VBD", "VBN", "VBP", "VBZ", "VBG"]
ignoreVerbBe = False

################################################################################
################################################################################


def get_substring_span(substring, string, start_index):
    """Get the span indexes of the substring.

    :param substring: Substring to find the span indexes.
    :param string: Text string.
    :param start_index: Offset to start searching for the substring.
    :return: End index, span info tuple.
    """
    start_ind = string.find(substring, start_index)
    end_ind = start_ind + len(substring)
    return end_ind, (start_ind, end_ind, substring)


def convert_spans_to_BIO(txt, tags):
    """Convert the spans to BIO format.

    :param txt: Text string.
    :param tags: List of tags.
    :return: List of text labels, List of all labels.
    """
    txt = txt.split(" ")
    text_label = [txt[0]]
    label = "O"
    flag = 1
    all_labels = []
    prev_label = ""

    # Convert gold tags into BIO format.
    for tok_id, val in enumerate(tags):
        if val[0] == "(":
            flag = 0
            label = "B-" + val[1:-1]
            if val[-1] == ")":
                text_label = [val[1:-2] + ": " + txt[tok_id]]
            else:
                text_label = [val[1:-1] + ": " + txt[tok_id]]

        if val[-1] == ")":
            flag = 1
            if label.startswith("I-"):
                label = label.replace("B-", "I-")
                text_label.append(txt[tok_id])
            elif label.startswith("B-") and prev_label[2:] == label[2:]:
                label = label.replace("B-", "I-")
                text_label.append(txt[tok_id])
            elif label.startswith("B-"):
                label = label[:-1]

            all_labels.append(text_label)
            prev_label = ""
            text_label = []
            continue

        if val[0] == "*":
            if flag == 1:
                label = "O"
                all_labels.append(txt[tok_id])
            elif label.startswith("B-"):
                flag = 0
                label = label.replace("B-", "I-")
                text_label.append(txt[tok_id])
            elif label.startswith("I-"):
                flag = 0
                label = label.replace("I-", "I-")

                text_label.append(txt[tok_id])
        prev_label = label

    return text_label, all_labels


def get_description(untok_txt, txt, tags):
    """Given the sentence text and list of span labels output argument spans.

    :param untok_txt: Un-tokenized text from the metadata.
    :param txt: Tokenized text, concatenation of the tokens.
    :param tags: List of tags for each token.
    :return: String of all the labels, list of the argument labels and spans.
    """

    text_label, all_labels = convert_spans_to_BIO(txt, tags)

    final_labels = []
    str_txt = ""
    start_ind = 0
    for lab in all_labels:
        if isinstance(lab, list):
            str_txt += "[" + " ".join(lab) + "] "
            arg_type = lab[0].split(": ")[0]
            arg_txt = " ".join(lab).split(": ")[1:]
            arg_span = ": ".join(arg_txt)
            start_ind, arg_span = get_substring_span(arg_span, untok_txt, start_ind)
            final_labels.append((arg_type, arg_span))
        else:
            str_txt += lab + " "
            start_ind, _ = get_substring_span(lab, untok_txt, start_ind)
    return str_txt, final_labels


def convert(tup, di):
    """Add the tuple values to the dictionary, convert tuple values to dict.

    :param tup: Tuple list.
    :param di: Dictionary to add the tuples.
    :return: Dictionary with the new values from the tuple.
    """
    for a, b in tup:
        di.setdefault(a, b)
    return di


def conllu_id_head_pairs(data):
    """Get token and head column as pairs.

    :param data: CoNLL-U block as SenChunk.
    :return: List of all pairs of token and head columns, list of token text.
    """
    all_pairs = []
    pairs = []
    txt = []
    tokens = []
    for sen in data.all_sen:
        for line in sen.sen:
            tok = line
            pairs.append((int(tok[0]), int(tok[6])))
            tokens.append(tok[1])
        all_pairs.append(pairs)
        txt.append(" ".join(tokens))
        pairs = []
        tokens = []
    return all_pairs, txt


def get_contiguous_index(span, head):
    """Find the sets of contiguous spans containing the head.

    :param span: List of spans.
    :param head: Head to find the span.
    :return: Span of the head.
    """

    label_blocks = []
    label_block = []
    for ii, lab in enumerate(span):
        if not label_block:
            label_block = [lab]
        else:
            if lab - label_block[-1] > 1:
                label_blocks.append(label_block)
                label_block = [lab]
            else:
                label_block.append(lab)
    label_blocks.append(label_block)

    # Find the span where the head is.
    for label_block in label_blocks:
        if head in label_block:
            break

    return label_block


def node_removal_constraints(label_block, sen, head, pred_id):
    """Remove nodes matching the constraints.

    Currently, just some constraints on values of DEPREL.
    Left comments to handle future cases.

    :param label_block: List of token_id to be part of the span.
    :param sen: CoNLL-U block as SenChunk.
    :param head: Token ID of the head.
    :return: Spans with nodes matching the constraints removed.
    """
    deprel_ = ["case"]
    pos_ = ["SCONJ"]

    span = set()  # Head should remain as part of span no matter what the pos
    span.add(head)

    remove_case = True
    remove_cop = True
    remove_extreme_puncts = True
    remove_relative_clause = True
    # if sen.tokens[sen.tokens[head - 1].head - 1].deprel == "acl":
    #     remove_case = True

    '''case1: '''
    for lab in label_block:
        # if remove_case:
        #     if (sen.tokens[lab-1].deprel not in deprel_):  # and (sen.tokens[lab-1].upos not in pos_ ):
        #         span.append(lab)
        if remove_case:
            if (sen.tokens[lab - 1].deprel not in deprel_):  # and (sen.tokens[lab-1].upos not in pos_ ):
                span.add(lab)
            if (sen.tokens[lab - 1].deprel in deprel_) and (sen.tokens[head - 1].deprel.startswith("obl")):
                if sen.tokens[sen.tokens[head - 1].head - 1].id == pred_id:
                    span.add(lab)
        # if remove_cop:
    span = list(span)
    span.sort()
    # span = get_contiguous_index(span, head)

    if remove_extreme_puncts:
        if span != []:
            if (sen.tokens[span[-1] - 1].deprel in ["punct", "P"]) and (sen.tokens[span[-1] - 1].lemma == ","):
                span.pop(-1)
        if span != []:
            if (sen.tokens[span[0] - 1].deprel in ["punct", "P"]) and (sen.tokens[span[-1] - 1].lemma == ","):
                span.pop(0)
    span.append(head)
    span.sort()
    # if remove_relative_clause:

    return span


def get_span(sen, pred, tree, head):
    """Compute the span from the head node.

    TODO: Add more details.

    :param sen: CoNLL-U block as SenChunk.
    :param pred: Tuple describing predicate.
    :param tree: dict keys:tok_ids, values:dep
    :param head: Token ID of the head.
    :return: List of spans.
    """

    remove_nsubj = False
    ind = -1
    for tok_id, token in enumerate(sen.tokens):
        if token.ispred:
            ind += 1
            if ind == int(pred[0]):
                break

    pred_id = tok_id + 1
    to_examine = collections.deque([head])
    span = []
    while len(to_examine) != 0:
        hea = to_examine.popleft()
        span.append(hea)
        indx = []

        for k, v in tree.items():
            if (v == hea) and (k != pred_id):
                indx.append(k)

        to_examine.extend(indx)
    if remove_nsubj:
        for tok_id in span:
            if sen.tokens[tok_id - 1].deprel == "nsubj":
                span.remove(tok_id)
    span.sort()

    # Get the contiguous span containing the head.
    span = get_contiguous_index(span, head)

    # Remove deprel_ and pos_ from the final span.
    span = node_removal_constraints(span, sen, head, pred_id)

    return span


def get_arg_span_block(sen, pred, arg_ind, id_head_dict, sen_map):
    """Get the argument spans for the given predicate, computed from the head.

    TODO: Add more details.

    :param sen: CoNLL-U block as SenChunk.
    :param pred: Given predicate, contains predicate info.
    :param arg_ind: Index of the column with the arguments for the given predicate.
    :param id_head_dict: Token ID of the head as a dict.
    :param sen_map:
    :return: List of argument spans.
    """
    args = []
    ind = 0
    for line in sen.sen:
        tok = line
        if tok[arg_ind] != "_":
            target_span = get_span(sen, pred, id_head_dict, int(tok[0]))
            target_span.sort()

            # Row info contains the span info at the end.
            start_tok, end_tok = tok[-2], tok[-1]

            # Take the first token, get the span from the row info.
            start_span = sen_map[target_span[0]][-2]  # start index of first token
            end_span = sen_map[target_span[-1]][-1]  # end index of last token
            arg_span_label = sen.txt[int(start_span):int(end_span)]
            args.append([sen.sen_id,
                         pred[0],
                         pred[1],
                         pred[1],
                         pred[2],
                         ind,
                         '(' + str(start_tok) + ',' + str(end_tok) + ',' + tok[1] + ')',
                         '(' + str(start_span) + ',' + str(end_span) + ',' + arg_span_label + ')',
                         tok[arg_ind]])
            ind += 1
    return args


def get_arg_block(sen, pred, arg_ind):
    """Get the argument spans for the given predicate, not computed.

    :param sen: CoNLL-U block as SenChunk.
    :param pred: Given predicate.
    :param arg_ind: Index of the column with the arguments for the given predicate.
    :return: List of arguments for the given predicate.
    """
    sent = []

    for line in sen.sen:
        sent.append(line)

    sent = list(map(list, zip(*sent)))
    argument_tags = sent[arg_ind]
    tok_txt = " ".join(sent[1])
    untok_txt = sen.txt
    str_txt, final_labels = get_description(untok_txt, tok_txt, argument_tags)
    ind = 0
    args = []
    for arg_id, arg in enumerate(final_labels):
        if arg[0] != "V":
            args.append([sen.sen_id,
                         pred[0],
                         pred[1],
                         pred[1],
                         pred[2],
                         ind,
                         '(' + str(arg[1][0]) + ',' + str(arg[1][1]) + ',' + arg[1][2] + ')',
                         '(' + str(arg[1][0]) + ',' + str(arg[1][1]) + ',' + arg[1][2] + ')',
                         arg[0]])
            ind += 1
    return args


def get_predicate_span(sen):
    """Get the predicates in the CoNLL-U block.

    Modifies the original function by just referencing the span info that's been
    injected into the CoNLL-U file.

    :param sen: CoNLL-U block as SenChunk.
    :return: List of predicates.
    """
    if len(sen[0]) < 10:
        logger.info("No predicates.")
        return []
    ind = 0
    predicates = []
    for tok in sen:
        tok_col = tok
        if tok_col[8] == "Y":
            start, end = tok_col[-2], tok_col[-1]
            predicates.append((ind, '(' + str(start) + ',' + str(end) + ',' + tok_col[1] + ')', tok_col[9], tok_col[3]))
            ind = ind + 1
    return predicates


def merge_arg(a, b):
    """Merge the arguments.

    :param a: Argument span.
    :param b: Argument span.
    :return: Combined span.
    """
    a = a.split(",")
    b = b.split(",")
    char_index = [int(a[0][1:]), int(a[1]), int(b[0][1:]), int(b[1])]
    min_char = min(char_index)
    max_char = max(char_index)
    if min_char == int(a[0][1:]):
        text = a[2][:-1] + " " + b[2][:-1]
    else:
        text = b[2][:-1] + " " + a[2][:-1]
    c = "({},{},{})".format(min_char, max_char, text)
    return c


def merge_c_args(args):
    """Merge the C-* arguments.

    TODO: Add more details.

    :param args: List of arguments.
    :return: None
    """
    remove_index = []
    arg_list = list(map(list, zip(*args)))[-1]
    for arg_index, arg in enumerate(args):
        if arg[-1].startswith(("C")):
            arg_type = "-".join(arg[-1].split("-")[1:])
            if arg_type in arg_list:
                main_arg_ind = arg_list.index(arg_type)
                args[main_arg_ind][-3] = merge_arg(args[main_arg_ind][-3], args[arg_index][-3])
                args[main_arg_ind][-2] = merge_arg(args[main_arg_ind][-2], args[arg_index][-2])
                remove_index.append(arg_index)
            else:
                args[arg_index][-1] = args[arg_index][-1][2:]
                arg_list[arg_index] = args[arg_index][-1]
    for index in sorted(remove_index, reverse=True):
        del args[index]


def merge_rarg(a, b):
    """Merge the spans of the R-* arguments.

    :param a: Argument span.
    :param b: Argument span.
    :return: Combined span.
    """
    a = a.split(",")
    b = b.split(",")
    char_index = [int(a[0][1:]), int(a[1]), int(b[0][1:]), int(b[1])]
    min_char = min(char_index)
    max_char = max(char_index)
    text = a[2][:-1] + " " + b[2][:-1]
    c = "({},{},{})".format(min_char, max_char, text)
    return c


def merge_r_args(args):
    """Merge the R-* arguments.

    TODO: Add more details.

    :param args: List of arguments.
    :return: None
    """
    arg_list = list(map(list, zip(*args)))[-1]
    for arg_index, arg in enumerate(args):
        if arg[-1].startswith(("R")):
            arg_type = "-".join(arg[-1].split("-")[1:])
            if arg_type in arg_list:
                main_arg_ind = arg_list.index(arg_type)
                if main_arg_ind > arg_index:
                    args[arg_index][-3] = merge_rarg(args[main_arg_ind][-3], args[arg_index][-3])
                    args[arg_index][-2] = merge_rarg(args[main_arg_ind][-2], args[arg_index][-2])
                    args[arg_index][-1] = args[main_arg_ind][-1] + args[arg_index][-1]
                else:
                    args[arg_index][-3] = merge_rarg(args[arg_index][-3], args[main_arg_ind][-3])
                    args[arg_index][-2] = merge_rarg(args[arg_index][-2], args[main_arg_ind][-2])
                    args[arg_index][-1] = args[arg_index][-1] + args[main_arg_ind][-1]
            else:
                args[arg_index][-1] = args[arg_index][-1] + "_"


def get_arguments_span(sen, predicate, id_head, compute_span=True):
    """Get the span of the arguments.

    This function modifies the original by taking the span info from the
    injected span index using a reference conllu file.

    :param sen: CoNLL-U blocks as SenChunks.
    :param predicate: List of predicates.
    :param id_head: Token ID of the head.
    :param compute_span: Whether to compute the span from the head or get from the data.
    :return: List of argument spans.
    """

    id_head_dict = convert(id_head, {})

    # Make a map of the parse tree token id.
    sen_map = {}
    for tok_line in sen.sen:
        ss = tok_line
        sen_map[int(ss[0])] = ss

    all_args = []
    for pred in predicate:
        if ignoreVerbBe:
            if pred[2].startswith('be.') or pred[2].startswith('being.'):
                continue
        arg_ind = 10 + pred[0]
        if compute_span:
            args = get_arg_span_block(sen, pred, arg_ind, id_head_dict, sen_map)
        else:
            args = get_arg_block(sen, pred, arg_ind)

        if args:
            merge_r_args(args)
            merge_c_args(args)
        all_args.extend(args)
    return all_args


def conllu2actionrole_df_span(conllu_filename, verb_predicate_only=False, compute_span=True):
    """Convert the CoNLL-U formatted file into CSVs containing the predicates
    and their arguments.

    :param conllu_filename: Path to the CoNLL-U file.
    :param verb_predicate_only: Restrict to predicates that are verbs.
    :param compute_span: Compute the span of the arguments.
    :return: Dataframes containing the predicates and the predicate-argument pairs.
    """
    sentence_data = []
    predicate_arguments = []
    predicates = []
    source_data = Reader(conllu_filename, "conllu")
    source_id_head_pairs, source_tokens = conllu_id_head_pairs(source_data)
    n_predicates = []
    test_sen = source_data.all_sen[0].sen
    arg_list = sum(list(map(list, zip(*test_sen)))[9:], [])
    can_compute_span = True
    if '*' in arg_list:
        can_compute_span = False
    if compute_span != can_compute_span:
        raise Exception("compute_span != can_compute_span")

    for ii, sen in enumerate(source_data.all_sen):
        predicate = []
        predicate_span = get_predicate_span(sen.sen)
        n_predicates.append(len(predicate_span))
        if predicate_span:
            all_args = get_arguments_span(sen, predicate_span, source_id_head_pairs[ii], compute_span=compute_span)
            sentence_data.append([sen.sen, source_tokens[ii]])

            for pred in predicate_span:
                if ignoreVerbBe:
                    if pred[2].startswith('be.') or pred[2].startswith('being.'):
                        continue
                if verb_predicate_only:
                    if pred[-1] in VBZ_names:
                        predicate.append([sen.sen_id, pred[0], pred[1], pred[1], pred[2]])
                else:
                    predicate.append([sen.sen_id, pred[0], pred[1], pred[1], pred[2]])

            if sen.sen_id == source_data.all_sen[ii - 1].sen_id:
                for arg_id, _ in enumerate(all_args):
                    all_args[arg_id][1] = all_args[arg_id][1] + n_predicates[ii - 1]
                for pred_id, _ in enumerate(predicate):
                    predicate[pred_id][1] = predicate[pred_id][1] + n_predicates[ii - 1]
                n_predicates[ii] = n_predicates[ii] + n_predicates[ii - 1]
            predicate_arguments.extend(all_args)
            predicates.extend(predicate)

    df_predicate_argument = pd.DataFrame(data=predicate_arguments,
                                 columns=["Document label", "aid", "verb", "verbSpan", "verbSense", "id", "roleHead",
                                          "roleSpan", "type"])

    df_predicate = pd.DataFrame(data=predicates,
                             columns=["Document label", "id", "verb", "span", "verbSense"])

    return df_predicate_argument, df_predicate
