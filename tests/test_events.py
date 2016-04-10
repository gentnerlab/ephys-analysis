import unittest
import datetime as dt
import numpy as np
from ephys import events


class EventTest(unittest.TestCase):

    def test_is_not_floatable(self):
        assert events._is_not_floatable(3.0)==False
        assert events._is_not_floatable('I love lamp')==True

    def test_is_correct(self):
        assert events.is_correct('F')==True
        assert events.is_correct('f')==True
        assert events.is_correct('T')==False
        assert np.isnan(events.is_correct(np.nan))

    def test_calc_rec_datetime(self):
        file_origin = '/mnt/cube/justin/matfiles/Pen01_Lft_AP2500_ML1350__Site04_Z1466__B997_cat_P01_S04_Epc01-03/SubB997Pen01Site04Epc02File01_10-25-15+12-46-08_B997_block.mat'
        start_time = 4.8e-05
        assert events.calc_rec_datetime(file_origin,start_time)==dt.datetime(2015, 10, 25, 12, 46, 8, 48)

def main():
    unittest.main()

if __name__ == '__main__':
    main()