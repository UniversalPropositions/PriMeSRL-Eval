# Tests of examples and results from the paper

We encoded the examples presented in the paper as unit tests in this folder.

## Test data folder structure

The test data have the following structure:
- `tests/data/<evaluation>/input` - input data for the tests.
    - Reference gold data always has `test` in the name (from the
      `train/valid/test` split naming scheme).
    - One test is with the gold predictions, containing `pred_gold`. This is a
      sanity check. All values should be `1` (i.e. `100%`).
    - Other predicted outputs have the same naming scheme as the paper (i.e.
      `p1`, `p2`, etc).
- `tests/data/<evaluation>/expected` - expected output data from the tests.
    - Contains the intermediate files for the evaluation scripts and the final
      comparison results.
    - Comparison results are in folders of this structure:
      `compare-<gold>-<predicted>/comparison-results-[official-conll|proposed].csv`.
- `tests/data/<evaluation>/output` - output folder for the tests, generated
    after running the tests and not committed.
    - This folder should have the same files and folder structure as the
      `expected` folder. Unit test will fail (and print out) on any differences
      in the files or folders of the tests.

Note that the test data for the `conll09` evaluations are in the top level
`data` folder as these are used as part of the main functions.

## Matching the test outputs with examples in the paper

Our evaluation scripts has more detailed outputs than the CoNLL2005 and
CoNLL2009 papers, and we have unified the naming scheme to help the reader
interpret and compare the numbers. We will run through an example for Table 5
and Table 1 in the paper.

### Table 5

To get the results for table 5, which is the CoNLL2009 data sets, run these
commands:
- (In-domain) `python run_evaluations.py -g data/conll09.wsj.test.conllu -p data/conll09.wsj.pred.roberta-base.conllu`
- (Out-of-domain) `python run_evaluations.py -g data/conll09.brown.test.conllu -p data/conll09.brown.pred.roberta-base.conllu`

We have the following formatted output from the main evaluation script for the
out-of-domain data:
```
********************************************************************************
Official CoNLL (2005 and 2009) evaluation:
         Metric          Type Precision  Recall      F1
0  conll09-head   PredicateId    99.92%  99.92%  99.92%
1  conll09-head     Predicate    89.83%  89.83%  89.83%
2  conll09-head  ArgumentHead    82.96%  83.08%  83.02%
3  conll05-span  ArgumentSpan    81.54%  77.26%  79.34%
********************************************************************************
********************************************************************************
Proposed evaluation:
     Metric            Type Precision  Recall      F1
0  proposed     PredicateId    99.92%  99.92%  99.92%
1  proposed       Predicate    83.32%  83.32%  83.32%
2  proposed    ArgumentHead    73.87%  74.26%  74.06%
3  proposed     CoreArgHead    74.74%  75.97%  75.35%
4  proposed  ContextArgHead    71.41%  69.63%  70.51%
********************************************************************************
```
The different types of evaluation are
- `PredicateId` - predicate identification.
- `Predicate` - predicate identification and sense disambiguation.
- `ArgumentHead` - head of the argument (both core arguments and contextual arguments).
- `ArgumentSpan` - span of the argument (no head).
- `CoreArgHead` - head of the core arguments (i.e. `A0`, `A1`, etc).
- `ContextArgHead` - head of the contextual arguments (i.e. `AM-TMP`, `AM-LOC`, etc).

So for table 5, out-of-domain results, the evaluation numbers are from row
`1` (`conll09-head` `Predicate`) and `2` (`conll09-head` `ArgumentHead`), which
is to be compared with the same rows (`1` and `2`) for our proposed evaluation.


### Table 1

To get the results for table 1, which are the results for the different
predicate sense, we look into the `tests/data/sense/compare-*` folders.
Particularly these two files:
- `comparison-results-official-conll.csv` - results from the official CoNLL evaluation.
    - E.g. For `P3`, run `python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p3/comparison-results-official-conll.csv`
- `comparison-results-proposed.csv` - results of our proposed evaluation.
    - E.g. For `P3`, run `python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p3/comparison-results-proposed.csv`

Note that the CSV files for the official CoNLL2009 evaluation scripts do not
provide the breakdown of the `Correct`, `Missing`, `Spurious` numbers. We have
`0` as placeholders in case we are able to modify the official scripts to
output those numbers in the future.

Example outputs for the `P1` (predication 1) results for table 1:
```
$ python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p3/comparison-results-official-conll.csv
         Metric          Type Precision   Recall       F1
0  conll09-head   PredicateId   100.00%  100.00%  100.00%
1  conll09-head     Predicate   100.00%  100.00%  100.00%
2  conll09-head  ArgumentHead   100.00%  100.00%  100.00%
3  conll05-span  ArgumentSpan     0.00%    0.00%    0.00%

$ python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p3/comparison-results-proposed.csv
     Metric            Type Precision   Recall       F1
0  proposed     PredicateId   100.00%  100.00%  100.00%
1  proposed       Predicate     0.00%    0.00%    0.00%
2  proposed    ArgumentHead    33.33%   33.33%   33.33%
3  proposed     CoreArgHead     0.00%    0.00%    0.00%
4  proposed  ContextArgHead   100.00%  100.00%  100.00%
```
The table results `Sense` `P` (precision) and `R` (recall) results correspond to
row `1` in both outputs above. Similar, row `2` in both outputs above
correspond to table results for `Args` `P` and `R`. Note that the paper
presents the fraction for the percentage results in the outputs above.

We can see the differences in the official evaluation and our proposed
evaluation. Please see Table 7 and its discussion for why these numbers are
different with additional examples.


### Tables 2 and 3

Please look at the `tests/data/sense-c` and `tests/data/sense-r` folders for
these results. Follow the descriptions for Table 1 above.
