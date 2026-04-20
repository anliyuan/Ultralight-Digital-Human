import unittest

import torch

from syncnet_training_utils import apply_training_step


def _loss(model):
    x = torch.tensor([[1.0]])
    y = torch.tensor([[1.0]])
    return (model(x) - y).pow(2).mean()


class SyncnetTrainingUtilsTests(unittest.TestCase):
    def test_apply_training_step_matches_explicit_zero_grad_behavior(self):
        helper_model = torch.nn.Linear(1, 1, bias=False)
        explicit_model = torch.nn.Linear(1, 1, bias=False)
        buggy_model = torch.nn.Linear(1, 1, bias=False)

        with torch.no_grad():
            helper_model.weight.fill_(0.0)
            explicit_model.weight.fill_(0.0)
            buggy_model.weight.fill_(0.0)

        helper_optimizer = torch.optim.SGD(helper_model.parameters(), lr=0.1)
        explicit_optimizer = torch.optim.SGD(explicit_model.parameters(), lr=0.1)
        buggy_optimizer = torch.optim.SGD(buggy_model.parameters(), lr=0.1)

        for _ in range(2):
            apply_training_step(_loss(helper_model), helper_optimizer)

            explicit_optimizer.zero_grad(set_to_none=True)
            explicit_loss = _loss(explicit_model)
            explicit_loss.backward()
            explicit_optimizer.step()

            buggy_loss = _loss(buggy_model)
            buggy_loss.backward()
            buggy_optimizer.step()

        self.assertAlmostEqual(
            helper_model.weight.item(), explicit_model.weight.item(), places=6
        )
        self.assertNotAlmostEqual(
            helper_model.weight.item(), buggy_model.weight.item(), places=6
        )


if __name__ == "__main__":
    unittest.main()
