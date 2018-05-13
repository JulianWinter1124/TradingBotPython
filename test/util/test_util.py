import pandas as pd
import numpy as np
import unittest

from util import data_enhancer


class DataEnhancerTest(unittest.TestCase):
    def test_make_binary_labels(self):
        test1 = [1, 2, 3, 4, 5, 6, 7, 8]
        test1_expect = [1, 1, 1, 1, 1, 1, 1]
        calc1 = data_enhancer.make_binary_labels(test1)
        self.assertEqual(test1_expect, calc1)

        test2 = [8, 7, 6, 5, 4, 3, 2, 1]
        test2_expect = [-1, -1, -1, -1, -1, -1, -1]
        calc2 = data_enhancer.make_binary_labels(test2)
        self.assertEqual(test2_expect, calc2)

        test3 = [8, 7, 6, 7, 4, 3, 2, 3]
        test3_expect = [-1, -1, 1, -1, -1, -1, 1]
        calc3 = data_enhancer.make_binary_labels(test3)
        self.assertEqual(test3_expect, calc3)
        print(calc3)

    def test_make_ranged_labels(self):
        test1 = [1, 2, 1, 3, 2, 2, 1]
        test1_expect = [0.5, -0.5, 1, -0.5, 0, -0.5]
        calc1 = data_enhancer.make_ranged_labels(test1)
        self.assertEqual(test1_expect, calc1)

        test3 = [1, 2, 3, 4, 5]
        test3_expect = [0.25, 0.25, 0.25, 0.25]
        calc3 = data_enhancer.make_ranged_labels(test3)
        self.assertEqual(test3_expect, calc3)