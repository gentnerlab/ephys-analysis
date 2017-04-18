import unittest
import tempfile
import shutil
import ephys.core as core

class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._block_path = tempfile.mkdtemp()
        cls._broken_block_path = tempfile.mkdtemp()
        cls._fexts = [".kwik", ".kwx", ".raw.kwd", ".phy", ".prm", ".prb", "_info.json", ".txt"]
        cls._tmp_files = []
        for fext in cls._fexts:
            with tempfile.NamedTemporaryFile(suffix=fext, dir=cls._block_path, delete=False) as f:
                cls._tmp_files.append(f.name)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._block_path)
        shutil.rmtree(cls._broken_block_path)

    def _test_find_(self, find_, fext):
        assert find_(self._block_path) == self._tmp_files[self._fexts.index(fext)]
        with self.assertRaises(AssertionError):
            find_(self._broken_block_path)

    def test_find_kwik(self):
        self._test_find_(core.find_kwik, ".kwik")
    def test_find_kwd(self):
        self._test_find_(core.find_kwd, ".raw.kwd")
    def test_find_kwx(self):
        self._test_find_(core.find_kwx, ".kwx")
    def test_find_prb(self):
        self._test_find_(core.find_prb, ".prb")
    def test_find_info(self):
        self._test_find_(core.find_info, "_info.json")

def main():
    unittest.main()

if __name__ == '__main__':
    main()