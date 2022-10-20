import os
import pandas as pd
import numpy as np


################################################################################
################################################################################

class ComparisonLabel:
    """Class names used in the comparison step.
    """
    correct = 'Correct'
    missing = 'Missing'
    spurious = 'Spurious'
    incorrect = 'Incorrect'
    samevaluesamespandiffhead = 'SameValueSameSpanDiffHead'
    samespandiffvalue = 'SameSpanDiffValue'
    sameheadsamevaluediffspan = 'SameHeadSameValueDiffSpan'
    samevaluediffspan = 'SameValueDiffSpan'


def split_span(span):
    """Parses span string in the form "(1,4,the)" into a tuple list
    (i.e. [1, 4, 'the']).

    :param span: Span string.
    :return: Tuple list representation of the span string.
    """
    if span is None or pd.isnull(span):
        return [-1, -1, '']

    b = int(span.split(',')[0][1:])
    e = int(span.split(',')[1])
    pre = '(' + str(b) + ',' + str(e) + ','
    t = span[span.find('(' + str(b) + ',' + str(e) + ',') + len(pre):len(span) - 1]

    # Begin index, end index, text.
    return [b, e, t]


def spans_overlap(span1, span2):
    """Checks for overlapping spans.

    :param span1: Span 1 to check overlap.
    :param span2: Span 2 to check overlap.
    :return: True if spans overlap.
    """

    if (span1 is None) or (span2 is None):
        return False

    [b1, e1, _] = split_span(span1)
    [b2, e2, _] = split_span(span2)

    return True if (b1 <= e2 and b2 <= e1) else False


def combine_spans(span1, span2):
    """Combine two given spans.

    Used for combining the spans of the arguments.
    TODO: must handle cases where span2 < span1, or they overlap / one contains the other.

    :param span1: Span 1 to combine.
    :param span2: Span 2 to combine.
    :return: Combined span of span 1 and 2.
    """

    if (span1 is None) or (span2 is None):
        return '(-1,-1,'')'

    [b1, e1, t1] = split_span(span1)
    [b2, e2, t2] = split_span(span2)

    if e1 < b1 or e2 < b2:
        return '(-1,-1,'')'

    spacer = '...' if (t1.find(' ') > 0 or t2.find(' ') > 0) else ' '

    if e1 < b2:
        text = t1.split(' ')[0] + spacer + t2.split(' ')[-1]
        return '(' + str(b1) + ',' + str(e2) + ',' + text + ')'

    if e2 < b1:
        text = t2.split(' ')[0] + spacer + t1.split(' ')[-1]
        return '(' + str(b2) + ',' + str(e1) + ',' + text + ')'

    return '(-1,-1,'')'


################################################################################
################################################################################


def predicate_identification(gold_predicate_csv, pred_predicate_csv):
    """Compare the predicate CSV files.

    Each predicate is either a perfect match or else it's spurious / missing.
    Comparing the predicates ('verb' field), not the predicate sense.

    :param gold_predicate_csv: Predicate CSV file with gold labels.
    :param pred_predicate_csv: Predicate CSV file with predicated labels.
    :return: Dataframe containing the comparison of the gold and predicated labels.
    """

    df_gold = pd.read_csv(gold_predicate_csv)
    df_pred = pd.read_csv(pred_predicate_csv)

    # Outer join, then assign labels for all combinations.
    df = pd.merge(df_gold, df_pred, on=['Document label', 'verb'], how='outer', suffixes=('_e', '_a'))

    df['comparison_label_auto'] = ''

    # Same predicates -> Correct.
    idx = ~pd.isna(df['id_e']) & ~pd.isna(df['id_a'])
    df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.correct

    # Predicted predicate has no matching gold predicate -> Spurious (predicate).
    idx = pd.isna(df['id_e']) & ~pd.isna(df['id_a'])
    df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.spurious

    # Gold predicate has no matching predicted predicate -> Missing (predicate).
    idx = ~pd.isna(df['id_e']) & pd.isna(df['id_a'])
    df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.missing

    df.rename(columns={'Document label': 'sentence_id'}, inplace=True)

    # Pack the gold and predicated predicate spans into a dataframe.
    df = df.reindex(columns=list(df) + ['e_pred_spanbegin', 'e_pred_spanend', 'e_pred_spantext',
                                        'a_pred_spanbegin', 'a_pred_spanend', 'a_pred_spantext'])

    # Rewrite the values of the split span rows.
    for index, row in df.iterrows():
        if pd.isnull(row['id_e']):
            df.loc[index, 'e_pred_spanbegin'] = -1
            df.loc[index, 'e_pred_spanend'] = -1
            df.loc[index, 'e_pred_spantext'] = ''
        else:
            [b, e, t] = split_span(row['verb'])
            df.loc[index, 'e_pred_spanbegin'] = b
            df.loc[index, 'e_pred_spanend'] = e
            df.loc[index, 'e_pred_spantext'] = t

        if pd.isnull(row['id_a']):
            df.loc[index, 'a_pred_spanbegin'] = -1
            df.loc[index, 'a_pred_spanend'] = -1
            df.loc[index, 'a_pred_spantext'] = ''
        else:
            [b, e, t] = split_span(row['verb'])
            df.loc[index, 'a_pred_spanbegin'] = b
            df.loc[index, 'a_pred_spanend'] = e
            df.loc[index, 'a_pred_spantext'] = t

    # Select the dataframe columns for output.
    df_out = df[
        ['sentence_id', 'e_pred_spanbegin', 'e_pred_spanend', 'e_pred_spantext', 'a_pred_spanbegin', 'a_pred_spanend',
         'a_pred_spantext', 'comparison_label_auto']]

    return df_out


def predicate_sense(gold_predicate_csv, pred_predicate_csv):
    """Compare the predicate sense CSV files.

    Each predicate sense is either correct or incorrect.
    Only comparing sense where predicate identification is completely correct

    :param gold_predicate_csv: Predicate CSV file with gold labels.
    :param pred_predicate_csv: Predicate CSV file with predicated labels.
    :return: Dataframe containing the comparison of the gold and predicated labels.
    """

    df_gold = pd.read_csv(gold_predicate_csv)
    df_pred = pd.read_csv(pred_predicate_csv)

    df = pd.merge(df_gold, df_pred, on=['Document label', 'verb'], how='inner', suffixes=('_e', '_a'))

    # Default comparison label -> Incorrect.
    df['comparison_label_auto'] = ComparisonLabel.incorrect

    # Find where predicate senses for the gold and predicated are the same -> Correct.
    idx = (df['verbSense_e'] == df['verbSense_a'])
    df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.correct

    # Pack the dataframe for output.
    df.rename(columns={'Document label': 'sentence_id', 'verbSense_e': 'e_predsense', 'verbSense_a': 'a_predsense'},
              inplace=True)

    df = df.reindex(columns=list(df) + ['pred_spanbegin', 'pred_spanend', 'pred_spantext'])

    # Rewrite values.
    for index, row in df.iterrows():
        [b, e, t] = split_span(row['verb'])
        df.loc[index, 'pred_spanbegin'] = b
        df.loc[index, 'pred_spanend'] = e
        df.loc[index, 'pred_spantext'] = t

    # Select columns for output.
    df_out = df[['sentence_id', 'pred_spanbegin', 'pred_spanend', 'pred_spantext', 'e_predsense', 'a_predsense',
                 'comparison_label_auto']]

    return df_out


################################################################################
################################################################################

def argument_identification(gold_arg_csv, pred_arg_csv, field_map, cond_predicate_location_and_sense=False):
    """Compare the argument (CoreArg or ContextArg) CSV files.

    Two types of evaluation, set `cond_predicate_location_and_sense`:
    - True -> Evaluation conditional on the correctness of the predicate location and predicate sense.
    - False -> Unconditional evaluation, ignore the correctness of the predicate predictions.

    :param gold_arg_csv: Argument (CoreArg or ContextArg) CSV file with gold labels.
    :param pred_arg_csv: Argument (CoreArg or ContextArg) CSV file with predicated labels.
    :param field_map: Column name mapping to standardize CoreArg and ContextArg column names.
    :param cond_predicate_location_and_sense: Whether to condition on predicate location and sense.
    :return: Dataframe containing the comparison of the gold and predicated labels.
    """

    df_e = pd.read_csv(gold_arg_csv)
    df_a = pd.read_csv(pred_arg_csv)

    if df_e.empty or df_a.empty:
        return pd.DataFrame()

    # Map the column names in the CSV file to the same column name.
    headfield = field_map['headfield']
    spanfield = field_map['spanfield']
    valuefield = field_map['valuefield']

    df_a['index'] = df_a.index
    df_e['index'] = df_e.index

    if cond_predicate_location_and_sense:
        df = pd.merge(df_e, df_a, on=['Document label', 'verb', 'verbSense'], how='outer', suffixes=('_e', '_a'))
    else:
        df = pd.merge(df_e, df_a, on=['Document label', 'verb'], how='outer', suffixes=('_e', '_a'))

    '''
    We construct the following fields to help generate the comparison labels.
        - field_anchor: the name of the field combination that matches, 'verb' field plus one or more other fields.
        - anchor_value: the value of the above field combination.
        - field_compare: the name of the field to be compared.
        - value_e: the expected value of the field to be compared.
        - value_a: the actual value of the field to be compared.
        - error_annotation: identifies potential source of error.
    '''

    df['comparison_label_auto'] = ''
    df['field_anchor'] = ''
    df['anchor_value'] = ''
    df['field_compare'] = ''
    df['value_e'] = ''
    df['value_a'] = ''
    df['error_annotation'] = ''

    ########################################
    # Spurious / missing verbs

    # These are the columns with empty indexes.

    idx_null_e = pd.isnull(df['index_e'])
    if not df.loc[idx_null_e].empty:
        df.loc[idx_null_e, 'comparison_label_auto'] = ComparisonLabel.spurious
        df.loc[idx_null_e, 'field_anchor'] = 'verb-' + valuefield
        df.loc[idx_null_e, 'anchor_value'] = df.loc[idx_null_e].apply(
            lambda x: combine_spans(x['verb'], x[spanfield + '_a'])[:-1], axis=1) + ' [' + df.loc[
                                                 idx_null_e, valuefield + '_a'] + '])'
        df.loc[idx_null_e, 'field_compare'] = 'verb'
        df.loc[idx_null_e, 'value_a'] = df.loc[idx_null_e, 'verb']
        df.loc[idx_null_e, 'additional_info'] = valuefield + '=' + df.loc[idx_null_e, valuefield + '_a']

    idx_null_a = pd.isnull(df['index_a'])
    if not df.loc[idx_null_a].empty:
        df.loc[idx_null_a, 'comparison_label_auto'] = ComparisonLabel.missing
        df.loc[idx_null_a, 'field_anchor'] = 'verb-' + valuefield
        df.loc[idx_null_a, 'anchor_value'] = df.loc[idx_null_a].apply(
            lambda x: combine_spans(x['verb'], x[spanfield + '_e'])[:-1], axis=1) + ' [' + df.loc[
                                                 idx_null_a, valuefield + '_e'] + '])'
        df.loc[idx_null_a, 'field_compare'] = 'verb'
        df.loc[idx_null_a, 'value_e'] = df.loc[idx_null_a, 'verb']
        df.loc[idx_null_a, 'additional_info'] = valuefield + '=' + df.loc[idx_null_a, valuefield + '_e']

    idx_notnull = (df['comparison_label_auto'] == '')

    # Dataframe with spurious / missing verbs
    df_null = df.loc[~idx_notnull]

    ########################################
    # Same verbs for both expected and actual values.
    # Compare the argument label value, head, and span.

    # These are the columns with non-empty indexes.

    df = df.loc[idx_notnull]

    # Add these boolean columns to help comparisons.
    if not df.empty:
        df['span_over'] = df.apply(lambda x: spans_overlap(x[spanfield + '_e'], x[spanfield + '_a']), axis=1)
        df['span_equal'] = df.apply(lambda x: x[spanfield + '_e'] == x[spanfield + '_a'], axis=1)
        df['head_equal'] = df.apply(lambda x: x[headfield + '_e'] == x[headfield + '_a'], axis=1)
        df['value_equal'] = df.apply(lambda x: x[valuefield + '_e'] == x[valuefield + '_a'], axis=1)
    else:
        df['span_over'] = None
        df['span_equal'] = None
        df['head_equal'] = None
        df['value_equal'] = None

    # Case 1: same span, same value, same head -> correct.
    idx = df['span_equal'] & df['value_equal'] & df['head_equal']
    if not df.loc[idx].empty:
        df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.correct
        df.loc[idx, 'error_annotation'] = 'correct'
        df.loc[idx, 'field_anchor'] = 'verb-' + spanfield + '-' + valuefield
        df.loc[idx, 'anchor_value'] = df.loc[idx].apply(lambda x: combine_spans(x['verb'], x[spanfield + '_e'])[:-1],
                                                        axis=1) + ' [' + df.loc[idx, valuefield + '_e'] + '])'
        df.loc[idx, 'field_compare'] = spanfield
        df.loc[idx, 'value_e'] = df.loc[idx, spanfield + '_e']
        df.loc[idx, 'value_a'] = df.loc[idx, spanfield + '_a']

    # Case 2: same span, same value, different head. Caused by parser difference.
    idx = df['span_equal'] & df['value_equal'] & ~df['head_equal']
    if not df.loc[idx].empty:
        df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.samevaluesamespandiffhead
        df.loc[idx, 'field_anchor'] = 'verb-' + valuefield + '-' + spanfield
        df.loc[idx, 'anchor_value'] = df.loc[idx].apply(lambda x: combine_spans(x['verb'], x[spanfield + '_e'])[:-1],
                                                        axis=1) + ' [' + df.loc[idx, valuefield + '_e'] + '])'
        df.loc[idx, 'field_compare'] = headfield
        df.loc[idx, 'value_e'] = df.loc[idx, headfield + '_e']
        df.loc[idx, 'value_a'] = df.loc[idx, headfield + '_a']
        df.loc[idx, 'error_annotation'] = 'parser_error'

    # Case 3: same span, different value.
    idx = df['span_equal'] & ~df['value_equal']
    if not df.loc[idx].empty:
        df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.samespandiffvalue
        df.loc[idx, 'field_anchor'] = 'verb-' + spanfield
        df.loc[idx, 'anchor_value'] = df.loc[idx].apply(lambda x: combine_spans(x['verb'], x[spanfield + '_e']), axis=1)
        df.loc[idx, 'field_compare'] = valuefield
        df.loc[idx, 'value_e'] = df.loc[idx, valuefield + '_e']
        df.loc[idx, 'value_a'] = df.loc[idx, valuefield + '_a']

    # Case 4: (overlapping span or different span), same value, same head.
    idx = df['span_over'] & ~df['span_equal'] & df['value_equal'] & df['head_equal']
    if not df.loc[idx].empty:
        df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.sameheadsamevaluediffspan
        df.loc[idx, 'field_anchor'] = 'verb-' + headfield + '-' + valuefield
        df.loc[idx, 'anchor_value'] = df.loc[idx].apply(lambda x: combine_spans(x['verb'], x[headfield + '_e'])[:-1],
                                                        axis=1) + ' [' + df.loc[idx, valuefield + '_e'] + '])'
        df.loc[idx, 'field_compare'] = spanfield
        df.loc[idx, 'value_e'] = df.loc[idx, spanfield + '_e']
        df.loc[idx, 'value_a'] = df.loc[idx, spanfield + '_a']

    # Case 5: none of the above, same value but different span.
    idx = df['value_equal'] & (df['comparison_label_auto'] == '')
    if not df.loc[idx].empty:
        df.loc[idx, 'comparison_label_auto'] = ComparisonLabel.samevaluediffspan
        df.loc[idx, 'field_anchor'] = 'verb-' + valuefield
        df.loc[idx, 'anchor_value'] = df.loc[idx].apply(lambda x: x['verb'][:-1], axis=1) + ' [' + df.loc[
            idx, valuefield + '_e'] + '])'
        df.loc[idx, 'field_compare'] = spanfield
        df.loc[idx, 'value_e'] = df.loc[idx, spanfield + '_e']
        df.loc[idx, 'value_a'] = df.loc[idx, spanfield + '_a']

    # Store the labeled cases 1 to 5.
    label_idx = (df['comparison_label_auto'] != '')
    df_labeled_1_5 = df.loc[label_idx]

    ########################################
    # Remaining cases with no comparison labels. Will be either missing or spurious arguments.

    # Missing argument -> anchor on verb
    idx_e_matched = df.loc[label_idx, 'index_e']
    idx = df['index_e'].isin(idx_e_matched)

    df_e_only_flag = False

    if False in idx.value_counts().keys():
        df_e_only_flag = True
        df_e_only = df.loc[~idx, ['Document label', 'verb', 'id_e', headfield + '_e', spanfield + '_e',
                                  valuefield + '_e', 'index_e']]
        if not df_e_only.empty:
            df_e_only.drop_duplicates(inplace=True)
            df_e_only['comparison_label_auto'] = ComparisonLabel.missing
            df_e_only['field_anchor'] = 'verb-' + valuefield
            df_e_only['anchor_value'] = df_e_only.apply(lambda x: combine_spans(x['verb'], x[spanfield + '_e'])[:-1],
                                                        axis=1) + ' [' + df_e_only[valuefield + '_e'] + '])'
            df_e_only['field_compare'] = spanfield
            df_e_only['value_e'] = df_e_only[spanfield + '_e']

    # Spurious argument -> anchor on verb
    idx_a_matched = df.loc[label_idx, 'index_a']
    idx = df['index_a'].isin(idx_a_matched)

    df_a_only_flag = False

    if False in idx.value_counts().keys():
        df_a_only_flag = True
        df_a_only = df.loc[~idx, ['Document label', 'verb', 'id_a', headfield + '_a', spanfield + '_a',
                                  valuefield + '_a', 'index_a']]
        if not df_a_only.empty:
            df_a_only.drop_duplicates(inplace=True)
            df_a_only['comparison_label_auto'] = ComparisonLabel.spurious
            df_a_only['field_anchor'] = 'verb-' + valuefield
            df_a_only['anchor_value'] = df_a_only.apply(lambda x: combine_spans(x['verb'], x[spanfield + '_a'])[:-1],
                                                        axis=1) + ' [' + df_a_only[valuefield + '_a'] + '])'
            df_a_only['field_compare'] = spanfield
            df_a_only['value_a'] = df_a_only[spanfield + '_a']

    ########################################
    # Build the final dataframe

    df = df_null  # Missing / spurious verbs
    df = pd.concat([df, df_labeled_1_5])  # Error cases
    if df_e_only_flag:
        df = pd.concat([df, df_e_only])  # Missing arguments
    if df_a_only_flag:
        df = pd.concat([df, df_a_only])  # Spurious arguments

    ########################################
    # Deduplication, in priority order of comparison label.

    # Firstly, we preprocess the rows to find the duplicate rows based on the
    # head, span, and value of the argument.

    # Find the duplicates in the actual dataframe.
    idx = (df.duplicated(subset=['verb', headfield + '_a', spanfield + '_a', valuefield + '_a'], keep=False)) & (
        ~df[headfield + '_a'].isna())
    df_dup_a = df.loc[
        idx, ['index_e', headfield + '_e', valuefield + '_e', 'index_a', headfield + '_a', valuefield + '_a',
              'comparison_label_auto']]
    df_dup_a['duplicate'] = 'actual'

    # Find the duplicates in the expected dataframe.
    idx = (df.duplicated(subset=['verb', headfield + '_e', spanfield + '_e', valuefield + '_e'], keep=False)) & (
        ~df[headfield + '_e'].isna())
    df_dup_e = df.loc[
        idx, ['index_e', headfield + '_e', valuefield + '_e', 'index_a', headfield + '_a', valuefield + '_a',
              'comparison_label_auto']]
    df_dup_e['duplicate'] = 'expected'

    # Combine the two dataframes above.
    df_dup = pd.concat([df_dup_e, df_dup_a])
    idx = (df_dup.duplicated(
        subset=['index_e', headfield + '_e', valuefield + '_e', 'index_a', headfield + '_a', valuefield + '_a',
                'comparison_label_auto']))

    # Remove the duplicates.
    df_dup = df_dup.loc[~idx]

    # For each comparison label, in priority order:
    #     - Elements that match with that label are accounted for, get their indexes.
    #     - Drop pair rows that include indexes of elements that have already been matched.
    # For the rest, convert them into 'Missing' and 'Spurious' rows.
    # Deduplicate again to remove any duplicates created as result of previous step.

    index_to_drop_list = []
    df_dup['keep'] = ''
    df_dup['dup_index'] = False

    # Labels in priority order for processing.
    comparison_labels = [ComparisonLabel.correct, ComparisonLabel.samevaluesamespandiffhead,
                         ComparisonLabel.samespandiffvalue, ComparisonLabel.sameheadsamevaluediffspan,
                         ComparisonLabel.samevaluediffspan]

    for lbl in comparison_labels:
        if df_dup.empty:
            break

        idx = df_dup['comparison_label_auto'] == lbl
        df_dup.loc[idx, 'keep'] = 'keep'

        idx_e_list = df_dup.loc[idx, 'index_e'].tolist()
        df_dup['dup_index'] = df_dup['index_e'].isin(idx_e_list)
        idx_to_drop = (df_dup['keep'] == '') & df_dup['dup_index']
        df_dup.loc[idx_to_drop, 'keep'] = 'drop'

        idx_a_list = df_dup.loc[idx, 'index_a'].tolist()
        df_dup['dup_index'] = df_dup['index_a'].isin(idx_a_list)
        idx_to_drop = (df_dup['keep'] == '') & df_dup['dup_index']
        df_dup.loc[idx_to_drop, 'keep'] = 'drop'

        # Store the indexes to drop.
        to_drop = df_dup[df_dup['keep'] == 'drop'].index.tolist()
        index_to_drop_list.extend(to_drop)
        df_dup = df_dup[df_dup['keep'] == '']

    # Check dataframe rows is paired with a 'keep' row, if not, convert to a
    # 'Missing' or 'Spurious' comparison label.

    df['keep'] = True
    df.loc[index_to_drop_list, 'keep'] = False

    idx_labeled = (df['keep'] == True)

    idx_e_labeled = df.loc[idx_labeled, 'index_e'].tolist()
    idx_a_labeled = df.loc[idx_labeled, 'index_a'].tolist()

    for i in index_to_drop_list:
        if (df.loc[i, 'index_e'] in idx_e_labeled) and (df.loc[i, 'index_a'] in idx_a_labeled):
            continue

        # Comparison label: Missing - Rows with no values on the expected side but have values on the actual side.
        if (df.loc[i, 'index_e'] not in idx_e_labeled) and (df.loc[i, 'index_a'] in idx_a_labeled):
            df.loc[i, 'comparison_label_auto'] = ComparisonLabel.missing
            cols = [headfield + '_a', spanfield + '_a', valuefield + '_a', 'value_a']
            for col in cols:
                df.loc[i, col] = ''
            cols = ['id_a', 'index_a']
            for col in cols:
                df.loc[i, col] = np.nan
            df.loc[i, 'field_anchor'] = 'verb-' + valuefield
            df.loc[i, 'anchor_value'] = combine_spans(df.loc[i, 'verb'], df.loc[i, spanfield + '_e']) + ' [' + df.loc[
                i, valuefield + '_e'] + '])'
            df.loc[i, 'field_compare'] = spanfield
            df.loc[i, 'value_e'] = df.loc[i, spanfield + '_e']
            df.loc[i, 'keep'] = True

        # Comparison label: Spurious - Rows with values on the expected side but no values on the actual side.
        if (df.loc[i, 'index_e'] in idx_e_labeled) and (df.loc[i, 'index_a'] not in idx_a_labeled):
            df.loc[i, 'comparison_label_auto'] = ComparisonLabel.spurious
            cols = [headfield + '_e', spanfield + '_e', valuefield + '_e', 'value_e']
            for col in cols:
                df.loc[i, col] = ''
            cols = ['id_e', 'index_e']
            for col in cols:
                df.loc[i, col] = np.nan
            df.loc[i, 'field_anchor'] = 'verb-' + valuefield
            df.loc[i, 'anchor_value'] = combine_spans(df.loc[i, 'verb'], df.loc[i, spanfield + '_a']) + ' [' + df.loc[
                i, valuefield + '_a'] + '])'
            df.loc[i, 'field_compare'] = spanfield
            df.loc[i, 'value_a'] = df.loc[i, spanfield + '_a']
            df.loc[i, 'keep'] = True

    # Filter out the values to keep.
    df = df[df['keep']]

    # One final deduplication in case some rows were missed.
    idx = (df.duplicated(
        subset=['Document label', 'verb', headfield + '_e', spanfield + '_e', valuefield + '_e', headfield + '_a',
                spanfield + '_a', valuefield + '_a', 'comparison_label_auto']))
    df = df[~idx]

    # Sort the dataframe.
    df.sort_values(by=['Document label', 'index_e', 'index_a', 'comparison_label_auto'], inplace=True)

    ########################################
    # Reformat dataframe to match desired schema.

    df.rename(columns={'Document label': 'sentence_id'}, inplace=True)
    df = df.reindex(columns=list(df) + ['pred_spanbegin', 'pred_spanend', 'pred_spantext',
                                        field_map['spanfield'] + '_e_spanbegin', field_map['spanfield'] + '_e_spanend',
                                        field_map['spanfield'] + '_e_spantext',
                                        field_map['spanfield'] + '_a_spanbegin', field_map['spanfield'] + '_a_spanend',
                                        field_map['spanfield'] + '_a_spantext',
                                        field_map['headfield'] + '_e_spanbegin', field_map['headfield'] + '_e_spanend',
                                        field_map['headfield'] + '_e_spantext',
                                        field_map['headfield'] + '_a_spanbegin', field_map['headfield'] + '_a_spanend',
                                        field_map['headfield'] + '_a_spantext',
                                        field_map['valuefield'] + '_e_value', field_map['valuefield'] + '_a_value'
                                        ])

    # Expand the span format into individual columns.
    for index, row in df.iterrows():
        [b, e, t] = split_span(row['verb'])
        df.loc[index, 'pred_spanbegin'] = b
        df.loc[index, 'pred_spanend'] = e
        df.loc[index, 'pred_spantext'] = t

        if not pd.isnull(row['id_e']):
            [b, e, t] = split_span(row[field_map['spanfield'] + '_e'])
            df.loc[index, field_map['spanfield'] + '_e_spanbegin'] = b
            df.loc[index, field_map['spanfield'] + '_e_spanend'] = e
            df.loc[index, field_map['spanfield'] + '_e_spantext'] = t

            [b, e, t] = split_span(row[field_map['headfield'] + '_e'])
            df.loc[index, field_map['headfield'] + '_e_spanbegin'] = b
            df.loc[index, field_map['headfield'] + '_e_spanend'] = e
            df.loc[index, field_map['headfield'] + '_e_spantext'] = t

            df.loc[index, field_map['valuefield'] + '_e_value'] = row[field_map['valuefield'] + '_e']

        if not pd.isnull(row['id_a']):
            [b, e, t] = split_span(row[field_map['spanfield'] + '_a'])
            df.loc[index, field_map['spanfield'] + '_a_spanbegin'] = b
            df.loc[index, field_map['spanfield'] + '_a_spanend'] = e
            df.loc[index, field_map['spanfield'] + '_a_spantext'] = t

            [b, e, t] = split_span(row[field_map['headfield'] + '_a'])
            df.loc[index, field_map['headfield'] + '_a_spanbegin'] = b
            df.loc[index, field_map['headfield'] + '_a_spanend'] = e
            df.loc[index, field_map['headfield'] + '_a_spantext'] = t

            df.loc[index, field_map['valuefield'] + '_a_value'] = row[field_map['valuefield'] + '_a']

    # Select the columns.
    df_out = df[['sentence_id', 'pred_spanbegin', 'pred_spanend', 'pred_spantext',
                 field_map['spanfield'] + '_e_spanbegin', field_map['spanfield'] + '_e_spanend',
                 field_map['spanfield'] + '_e_spantext',
                 field_map['headfield'] + '_e_spanbegin', field_map['headfield'] + '_e_spanend',
                 field_map['headfield'] + '_e_spantext',
                 field_map['valuefield'] + '_e_value',
                 field_map['spanfield'] + '_a_spanbegin', field_map['spanfield'] + '_a_spanend',
                 field_map['spanfield'] + '_a_spantext',
                 field_map['headfield'] + '_a_spanbegin', field_map['headfield'] + '_a_spanend',
                 field_map['headfield'] + '_a_spantext',
                 field_map['valuefield'] + '_a_value',
                 'comparison_label_auto']]

    return df_out


################################################################################
################################################################################


def compare_conllu_csv(gold_path, pred_path, output_folder):
    """Compare the CSV files of the exported CoNLL-U files.

    :param gold_path: Path of the gold CSV file.
    :param pred_path: Path of the predicted CSV file.
    :param output_folder: Folder to output the comparisons.
    :return: None
    """
    gold_action_csv = os.path.join(gold_path, 'Predicate.csv')
    pred_action_csv = os.path.join(pred_path, 'Predicate.csv')

    gold_role_csv = os.path.join(gold_path, 'PredicateCoreArgs.csv')
    pred_role_csv = os.path.join(pred_path, 'PredicateCoreArgs.csv')

    gold_context_csv = os.path.join(gold_path, 'PredicateContextArgs.csv')
    pred_context_csv = os.path.join(pred_path, 'PredicateContextArgs.csv')

    df_pred_id = predicate_identification(gold_action_csv, pred_action_csv)
    df_pred_id.to_csv(os.path.join(output_folder, 'CompPredId.csv'), index=False)

    df_pred_sense = predicate_sense(gold_action_csv, pred_action_csv)
    df_pred_sense.to_csv(os.path.join(output_folder, 'CompPredSense.csv'), index=False)

    field_map = {'headfield': 'roleHead', 'spanfield': 'roleSpan', 'valuefield': 'type'}
    cond_pred_loc_sense = True
    df_role_id = argument_identification(gold_role_csv, pred_role_csv, field_map, cond_pred_loc_sense)
    df_role_id.to_csv(os.path.join(output_folder, 'CompRoleArgIdLabel.csv'), index=False)

    field_map = {'headfield': 'contextHead', 'spanfield': 'contextSpan', 'valuefield': 'type'}
    df_context_id = argument_identification(gold_context_csv, pred_context_csv, field_map)
    df_context_id.to_csv(os.path.join(output_folder, 'CompContextArgIdLabel.csv'), index=False)
