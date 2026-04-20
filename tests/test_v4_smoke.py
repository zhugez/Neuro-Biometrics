import sys
import unittest
from pathlib import Path

import torch
from pytorch_metric_learning.losses import MultiSimilarityLoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from shared.model_multimodal import create_multimodal_model


class V4SmokeTest(unittest.TestCase):
    @staticmethod
    def _has_finite_grad(module) -> bool:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                return False
            if parameter.grad.abs().sum().item() > 0:
                return True
        return False

    def test_synthetic_forward_backward(self):
        torch.manual_seed(0)
        batch_size = 4
        noisy = torch.randn(batch_size, 4, 800)
        clean = torch.randn(batch_size, 4, 800)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        spectrogram = torch.randn(batch_size, 4, 65, 13)
        model = create_multimodal_model(
            backbone="resnet18",
            n_channels=4,
            embed_dim=128,
            pretrained=False,
            use_mamba=False,
            spec_embed_dim=128,
        )
        model.train()
        metric_loss = MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        denoised, fused = model(noisy, spectrogram)
        loss = metric_loss(fused, labels) + 0.01 * (denoised - clean).pow(2).mean()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(self._has_finite_grad(model.denoiser))
        self.assertTrue(self._has_finite_grad(model.embedder))
        self.assertTrue(self._has_finite_grad(model.spec_embedder))
        self.assertTrue(self._has_finite_grad(model.fusion))


if __name__ == "__main__":
    unittest.main()
