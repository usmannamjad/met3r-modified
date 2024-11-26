import torch
from met3r import MET3R
import unittest


class MET3RTest(unittest.TestCase):

    def setUp(self):
        self.metric = MET3R().cuda()

    def random_inputs(self):
        inputs = torch.randn((10, 2, 3, 256, 256)).cuda()
        inputs = inputs.clip(-1, 1)
        score, mask = self.metric(inputs)

        self.assertTrue(0.3 <= score <= 0.35)



if __name__ == '__main__':
    unittest.main()