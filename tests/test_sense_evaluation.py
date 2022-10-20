
import os
import unittest

from src.pipeline import proposed_evaluation, official_conll_evaluation
from tests.utils import get_test_folders, compare_folders


TEST_DIR = os.path.dirname(os.path.abspath(__file__))

################################################################################
################################################################################


class TestSenseEvaluation(unittest.TestCase):

    def setUp(self):
        self.test_folder = os.path.join(TEST_DIR, 'data')

    def test_eval_sense(self):
        expected_folder, input_folder, output_folder = get_test_folders(self.test_folder, 'sense')

        gold_conllu_fp = os.path.join(input_folder, 'sense_test.conllu')

        for m in ['gold'] + ['p'+str(i) for i in list(range(1, 4))]:
            pred_conllu_fp = os.path.join(input_folder, 'sense_pred_'+m+'.conllu')
            official_conll_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)
            proposed_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)

        compare_folders(expected_folder, output_folder)

    def test_eval_sense_c(self):
        expected_folder, input_folder, output_folder = get_test_folders(self.test_folder, 'sense-c')

        gold_conllu_fp = os.path.join(input_folder, 'sense_c_test.conllu')

        for m in ['gold'] + ['p'+str(i) for i in list(range(1, 8))]:
            pred_conllu_fp = os.path.join(input_folder, 'sense_c_pred_'+m+'.conllu')
            official_conll_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)
            proposed_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)

        compare_folders(expected_folder, output_folder)

    def test_eval_sense_r(self):
        expected_folder, input_folder, output_folder = get_test_folders(self.test_folder, 'sense-r')

        gold_conllu_fp = os.path.join(input_folder, 'sense_r_test.conllu')

        for m in ['gold'] + ['p'+str(i) for i in list(range(1, 7))]:
            pred_conllu_fp = os.path.join(input_folder, 'sense_r_pred_'+m+'.conllu')
            official_conll_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)
            proposed_evaluation(gold_conllu_fp, pred_conllu_fp, output_folder)

        compare_folders(expected_folder, output_folder)



################################################################################
################################################################################


if __name__ == '__main__':
    unittest.main()
