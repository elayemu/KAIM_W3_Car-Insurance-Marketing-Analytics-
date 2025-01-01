
# tests/test_data_loader.py

import unittest
from src.data_loader import load_data, check_missing_values
import pandas as pd

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        df = load_data("E:\\DS+ML\\AIM3\\Week3\\Data\\MachineLearningRating_v3.txt")
        self.assertIsInstance(df, pd.DataFrame)

    def test_check_missing_values(self):
        df = load_data("E:\\DS+ML\\AIM3\\Week3\\Data\\MachineLearningRating_v3.txt")
        missing_values = check_missing_values(df)
        self.assertGreater(len(missing_values), 0)
