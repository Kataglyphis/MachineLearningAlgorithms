import unittest

class TestMethods(unittest.TestCase):
    def test_add(self):
        self.assertEqual(":)", ":)")

    def test_add_again(self):
        self.assertEqual(":)", ":)")

    def test_if(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()