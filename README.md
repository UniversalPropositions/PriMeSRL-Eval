# PriMeSRL-Eval: A Practical Quality Metric for Semantic Role Labeling Systems Evaluation

This repository contains the code for the paper:
- `PriMeSRL-Eval: A Practical Quality Metric for Semantic Role Labeling Systems Evaluation`

We have our proposed evaluation of SRL quality and the official evaluation
from the CoNLL2005 (span evaluation) and CoNLL2009 (head evaluation). We use an
SRL format that extends the CoNLL-U format (see SRL format section below).

We have the following pipelines:
- Proposed evaluation: uses our extended CoNLL-U+SRL format.
- CoNLL2005 evaluation: CoNLL-U conversion to CoNLL2005 format for the official scripts.
- CoNLL2009 evaluation: CoNLL-U conversion to CoNLL2009 format for the official scripts.

### Data
- Data sources:
    - CoNLL2005: https://catalog.ldc.upenn.edu/LDC99T42 and https://www.cs.upc.edu/~srlconll/soft.html
    - CoNLL2009: https://catalog.ldc.upenn.edu/LDC2012T04 and https://ufal.mff.cuni.cz/conll2009-st/eval-data.html
- Download the data from LDC and store under `/data` repository
- We support reading data in CoNLL09, CoNLLUS and CONLLu formats. Any format get converted into our
CoNLL-U format for futher processing. We will provide code for conversion from CoNLL05 to our
CoNLL-U format so that the community can use our proposed evaluation and compare
to the official evaluations.


## Usage

- (Recommended) Create a virtual env, e.g.
    - `conda create -n eval python=3.9`
    - `conda activate eval`
- Install requirements: `pip install -r requirements.txt`
- (Optional) Verify unit tests (takes about 2 mins): `pytest tests`
- Run evaluation script:
    - `python run_evaluations.py --gold-conllu <file> --pred-conllu <file> --format <str>--output-folder <folder>`
      - `--format` can take following values: `[conll09, conll05, conllu]`
    - See the `data` and `tests/data` folder for examples.
    - Example usages:
      - `python run_evaluations.py -g data/conll09/gold_file -p data/conll09/pred_file -f conll09 -o tmp`

The evaluation script will show the results from the official CoNLL scripts and
our proposed evaluation method. Please see the paper on how to interpret and
compare these numbers.


## Examples from our paper

We have encoded all examples in our paper as unit tests. See `tests/README.md`
for how to match up numbers in the tests with those presented in the paper.

In short, the data for the tables are in the `tests/data/<evaluation>/input`
with similar naming scheme to the examples in the table. The evaluation results
presented in the paper are in these folders with this structure:
`tests/data/<evaluation>/expected/compare-*/comparison-results-[official-conll|proposed].csv`.

We provide a script to format these comparison results, example usages:
- `python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p1/comparison-results-official-conll.csv`
- `python format_results.py -c tests/data/sense/expected/compare-sense_test-sense_pred_p1/comparison-results-proposed.csv`


## CoNLL-U+SRL format

We use an extended [CoNLL-U format](https://universaldependencies.org/format.html)
that replaces the `MISC` column with the additional columns below:
- `ISPRED` - Flag `Y` when the token is a predicate, `_` otherwise.
- `PREDSENSE` - Predicate sense from [PropBank](https://propbank.github.io/).
- `ARGS` - One column for each predicate, in order of appearance, i.e. first
           argument column contains the arguments for the first predicate.


## Contribution

To contribute to this repository, particularly new unit tests of other
interesting error combinations, please open an  issue for discussion and any
subsequent PR.


## Citation

```bibtex
@misc{jindal2022primesrleval,
    title={PriMeSRL-Eval: A Practical Quality Metric for Semantic Role Labeling Systems Evaluation},
    author={Ishan Jindal and Alexandre Rademaker and Khoi-Nguyen Tran and Huaiyu Zhu and Hiroshi Kanayama and Marina Danilevsky and Yunyao Li},
    year={2022},
    eprint={2210.06408},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

