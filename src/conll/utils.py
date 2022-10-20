import logging

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def get_sen_tok(sen):
    """Convert CoNLL-U block from list of lines to list of list.

    From ["1\tHe\tPRP\t...", ...] to [[1, He, PRP, ...], ...].

    :param sen: CoNLL-U block as a list of lines.
    :return: CoNLL-U block with the lines converted to a list of values.
    """
    tok = []
    for tok_id in sen:
        tok.append(tok_id.strip().split("\t"))
    return tok


def get_tok_sen(tok):
    """Convert CoNLL-U block from list of list to list of lines.

    From [[1, He, PRP, ...], ...] to ["1\tHe\tPRP\t...", ...].

    :param tok: CoNLL-U block as a list of list of values.
    :return: CoNLL-U block as a list of lines.
    """
    sen = []
    for tok_id in tok:
        sen.append("\t".join(tok_id) + "\n")
    return sen


def get_predicate(sen):
    """Get the predicates from the CoNLL-U block.

    :param sen: CoNLL-U block as a list of list of values.
    :return: List of predicates in this form: [(predicate location, predicate verb, predicate.sense, predicate POS), (),()...]
    """
    if len(sen[0]) < 10:
        logger.debug("No predicate.")
        return []
    ind = 0
    predicates = []
    for tok in sen:
        tok_col = tok.strip().split("\t")
        if tok_col[8] == "Y":
            predicates.append((tok_col[0], tok_col[9].split(".")[0], tok_col[9], tok_col[3]))

            ind = ind + 1
    return predicates


def get_common_predicate(source_sen, target_sen, source_pred, target_pred):
    """Check for common predicates and remove the differing predicates and the
    corresponding arguments from both source and target sentences.

    :param source_sen: CoNLL-U block as a list of lines.
    :param target_sen: CoNLL-U block as a list of lines.
    :param source_pred: List of predicates.
    :param target_pred: List of predicates.
    :return: Source and target CoNLL-U blocks with common predicates, count of missing and spurious predicates, count of missing and spurious arguments
    """

    # Count the number of arguments associated with missing and spurious predicates.
    conditional_arg = {"missing": 0, "spurious": 0}

    source_col = list(zip(*get_sen_tok(source_sen)))
    target_col = list(zip(*get_sen_tok(target_sen)))

    source_tok_id = [int(x) for x in list(zip(*source_pred))[0]]
    target_tok_id = [int(x) for x in list(zip(*target_pred))[0]]
    common = list(set(target_tok_id) & set(source_tok_id))

    missing = set(source_tok_id) - set(common)
    if missing:
        for miss in missing:
            for tok in source_col[10 + source_tok_id.index(miss)]:
                if tok != "_":
                    conditional_arg["missing"] += 1
    spurious = set(target_tok_id) - set(common)

    if spurious:
        for miss in spurious:
            for tok in target_col[10 + target_tok_id.index(miss)]:
                if tok != "_":
                    conditional_arg["spurious"] += 1
    common.sort()
    source_index = []
    target_index = []
    for index in common:
        source_index.append(source_tok_id.index(index))
        target_index.append(target_tok_id.index(index))

    source_arg_col = [list(x) for x in source_col[8:]]
    target_arg_col = [list(x) for x in target_col[8:]]
    temp = list(set(source_tok_id) - set(common))

    for ind in temp:
        source_arg_col[0][ind - 1] = "_"
        source_arg_col[1][ind - 1] = "_"
    source_arg_col = source_arg_col[:2] + [source_arg_col[source_tok_id.index(ind) + 2] for ind in common]
    temp = list(set(target_tok_id) - set(common))

    for ind in temp:
        target_arg_col[0][ind - 1] = "_"
        target_arg_col[1][ind - 1] = "_"
    target_arg_col = target_arg_col[:2] + [target_arg_col[target_tok_id.index(ind) + 2] for ind in common]
    source_col = source_col[:8] + source_arg_col
    target_col = target_col[:8] + target_arg_col
    source_sen = get_tok_sen(zip(*source_col))
    target_sen = get_tok_sen(zip(*target_col))

    return source_sen, target_sen, len(missing), len(spurious), conditional_arg


def check_predicate_pos(gold_sen, pred_sen, syn=False):
    """Output the CoNLL-U blocks with the common predicate location between the
    gold and predicted CoNLL-U files.

    :param gold_sen: List of CoNLL-U blocks in the gold file.
    :param pred_sen: List of CoNLL-U blocks in the predicted file.
    :param syn: Whether to evaluate syntactic dependency via official CoNLL scripts.
    :return: List of common predicates, PRF metrics, and count of spurious and missing arguments.
    """
    assert len(gold_sen) == len(pred_sen)

    # Count the number of arguments associated with missing and spurious predicates.
    argument_count = {"missing": 0, "spurious": 0}

    new_gold = []
    new_pred = []
    pred_missing = 0
    pred_spurious = 0
    total_gold_pred = 0
    total_pred_pred = 0
    arg_missing = 0
    arg_spurious = 0

    logger.debug("Total number of sentences: %d", len(pred_sen))

    for sen_id in range(len(pred_sen)):
        assert len(gold_sen[sen_id]) == len(pred_sen[sen_id])
        gold_pred = get_predicate(gold_sen[sen_id])
        pred_pred = get_predicate(pred_sen[sen_id])
        if gold_pred == [] and pred_pred != []:
            pred_spurious += len(pred_pred)
            for tok_col in list(map(list, zip(*get_sen_tok(pred_sen[sen_id]))))[10:]:
                for tok in tok_col:
                    if tok != "_":
                        arg_spurious += 1
            continue
        if gold_pred != [] and pred_pred == []:
            pred_missing += len(gold_pred)
            for tok_col in list(map(list, zip(*get_sen_tok(gold_sen[sen_id]))))[10:]:
                for tok in tok_col:
                    if tok != "_":
                        arg_missing += 1
            continue
        if len(gold_pred) == 0 and len(pred_pred) == 0:
            new_gold.append(gold_sen[sen_id])
            new_pred.append(pred_sen[sen_id])
            if not syn:
                continue

        gold_mod_sen, pred_mod_sen, missing, spurious, conditional_arg = get_common_predicate(gold_sen[sen_id], pred_sen[sen_id], gold_pred, pred_pred)
        gold_pred = get_predicate(gold_mod_sen)
        pred_pred = get_predicate(pred_mod_sen)
        new_gold.append(gold_mod_sen)
        new_pred.append(pred_mod_sen)

        pred_missing = pred_missing + missing
        pred_spurious = pred_spurious + spurious
        total_gold_pred += len(gold_pred)
        total_pred_pred += len(pred_pred)
        argument_count["missing"] += conditional_arg["missing"]
        argument_count["spurious"] += conditional_arg["spurious"]
    argument_count["missing"] += arg_missing
    argument_count["spurious"] += arg_spurious

    logger.debug("Total number of predicates this evaluation is on: %d", total_gold_pred)
    logger.debug("Missing Predicates: %d", pred_missing)
    logger.debug("Spurious Predicates: %d", pred_spurious)

    tp = total_gold_pred
    fp = pred_spurious
    fn = pred_missing
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r)

    logger.debug("PID: p: %.2f, r: %.2f, f1: %.2f", p, r, f1)
    return new_gold, new_pred, (tp, fp, fn, p, r, f1), argument_count

################################################################################
################################################################################


def combine_performance_scores(a, b):
    """Combine the quality scores of two sets of evaluation numbers.

    Assumed format: [Correct, Spurious, Missing, Precision, Recall, F1]

    :param a: Set of quality scores.
    :param b: Set of quality scores.
    :return: Combined set of quality scores.
    """
    c = [0, 0, 0, 0, 0, 0]

    # Correct, Spurious, Missing values
    for i in range(3):
        c[i] = a[i] + b[i]

    # Precision and Recall values
    for i in range(3, 5):
        c[i] = a[i] * b[i]

    # Recalcuate F1 from the new P R values.
    c[5] = (2 * c[3] * c[4]) / (1e-10 + c[3] + c[4])

    return c
