import os
import csv
import logging
from src.proposed.conllu_to_csv import conllu2actionrole_df_span
from src.proposed.utils import Reader, output_df

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def get_tokens(parsed):
    """Get the tokens in the parsed CoNLL-U block.

    :param parsed: CoNLL-U block.
    :return: List of token values from the CoNLLU block.
    """

    tokens = []
    for p in parsed:
        tokens.append(p[1])  # Pos 1 is the token. Pos 0 is the token id.
    return tokens


def find_token_spans(sentence, tokens, a_offset):
    """Finds the token span given a reference sentence.

    Start a particular offset for sentence split by the parser.

    :param sentence: CoNLL-U block of the sentence.
    :param tokens: Tokens to find the spans.
    :param a_offset: Calculate from the given offset (i.e. for same tokens).
    :return: Spans for each token in form (token name, left index, offset).
    """

    token_spans = []
    offset = a_offset
    for token in tokens:
        lidx = sentence.find(token, offset)
        if lidx <= -1:
            print("Bug -", "token:", token, "| offset:", offset, "| sentence:", sentence)
        assert (lidx > -1)
        ridx = lidx + len(token) - 1
        offset = ridx + 1
        token_spans.append([token, lidx, offset])
    return token_spans


def add_token_spans(sents_e, sents_a):
    """For each block in the actual (e.g. predicted) conllu file, find the span
    of each token using the expected (e.g. gold) sentences as reference.

    :param sents_e: CoNLL-U sentence blocks for the expected (e.g. gold) file.
    :param sents_a: CoNLL-U sentence blocks for the actual (e.g. predicted) file.
    :return:
    """
    token_spans_sents = []
    a_offset = 0
    a_prev_id = -1
    for sen_id, a in enumerate(sents_a):
        if a.meta.sen_id != "":
            a_curr_id = int(a.meta.sen_id)
        else:
            a_curr_id = sen_id

        if a_curr_id != a_prev_id:
            a_offset = 0
        tokens = get_tokens(a.sen)

        token_spans_sent = find_token_spans(sents_e[a_curr_id].txt, tokens, a_offset)
        for (t1, t2) in zip(a.sen, token_spans_sent):
            assert (t1[1] == t2[0])
            # t1.insert(2, t2[1])
            # t1.insert(3, t2[2])
            t1.append(t2[1])  # left index
            t1.append(t2[2])  # right index
            a_offset = t2[2]
        token_spans_sents += token_spans_sent
        a_prev_id = a_curr_id
    return token_spans_sents


def transfer_token_span_info(fn_e, fn_a):
    """Output intermediate file (CoNLL-U + 2 columns for span for each token)
    which is actual conllu file with the span information of the expected
    conllu file.

    :param fn_e: Filename of the expected (e.g. gold) CoNLL-U file
    :param fn_a: Filename of the actual (e.g. predicted) CoNLL-U file
    :return: Path of the intermediate file (fn_a with appended '.span' extension)
    """

    sents_e = Reader(fn_e, data_format="conllu").all_sen
    sents_a = Reader(fn_a, data_format="conllu").all_sen

    tokenSpans = add_token_spans(sents_e, sents_a)

    outfn = fn_a + ".span"
    with open(outfn, 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar='')
        for sen_id, sents in enumerate(sents_a):

            if sents.meta.sen_id == "":
                f.write("# {} # {}".format(sen_id, sents.meta.sen_txt) + "\n")
            else:
                f.write("# {} # {}".format(sents.meta.sen_id, sents.meta.sen_txt) + "\n")
            for row in sents.sen:
                writer.writerow(row)
            writer.writerow([])
    return outfn


def transfer_span_info(gold_file, pred_file, verb_predicate_only=False, compute_span=True):
    """Transfer the labels from gold file to the given file (conceptually).

    Write the results as CSV files: Action.csv, ActionRole.csv, and ActionContext.csv.
    The results will have spans from the gold file and labels from the given file.

    :param gold_file: Gold CoNLL-U file with parse+SRL
    :param pred_file: Predicted CoNLL-U file with parse+SRL
    :param verb_predicate_only: Only consider verb predicates.
    :param compute_span: Compute the span of the tokens.

    :return: None, but outputs files to the directory of the pred_file.
    """

    output_path = os.path.dirname(pred_file)
    os.makedirs(output_path, exist_ok=True)

    predicate_filename = os.path.join(output_path, "Predicate.csv")
    core_args_filename = os.path.join(output_path, "PredicateCoreArgs.csv")
    context_args_filename = os.path.join(output_path, "PredicateContextArgs.csv")

    out_filename = transfer_token_span_info(gold_file, pred_file)
    semantic_roles, predicates = conllu2actionrole_df_span(out_filename, verb_predicate_only=verb_predicate_only,
                                                           compute_span=compute_span)

    core_args = semantic_roles[semantic_roles.type.str.contains(r'\d')]
    context_args = semantic_roles[semantic_roles.type.str.contains('AM')]

    predicates.to_csv(predicate_filename, index=False)
    output_df(predicates, predicate_filename)
    core_args.to_csv(core_args_filename, index=False)
    output_df(core_args, core_args_filename)

    context_args = context_args.rename(columns={'roleHead': 'contextHead', 'roleSpan': 'contextSpan'})
    context_args.to_csv(context_args_filename, index=False)
    output_df(context_args, context_args_filename)
