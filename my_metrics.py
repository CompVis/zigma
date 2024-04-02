from einops import rearrange
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from utils.torchmetric_fdd import FrechetDinovDistance
from utils.torchmetric_fvd import FrechetVideoDistance
from utils.torchmetric_prdc import PRDC
from utils.torchmetric_sfid import sFrechetInceptionDistance
import torch.nn.functional as F


class MyMetric:
    def __init__(self, device="cuda", choices=["fid"]):
        self.choices = choices
        if "fid" in choices:
            self._fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True,
            ).to(device)
        if "is" in choices:
            self._is = InceptionScore().to(device)
        if "kid" in choices:
            self._kid = KernelInceptionDistance(subset_size=50).to(device)
        if "prdc" in choices:
            self._prdc = PRDC(nearest_k=5).to(device)
        if "sfid" in choices:
            self._sfid = sFrechetInceptionDistance().to(device)
        if "fdd" in choices:
            self._fdd = FrechetDinovDistance().to(device)
        if "fvd" in choices:
            self._fvd = FrechetVideoDistance().to(device)

    def update_real(self, data):
        if "fid" in self.choices:
            self._fid.update(data, real=True)
        if "is" in self.choices:
            self._is.update(data)
        if "kid" in self.choices:
            self._kid.update(data, real=True)
        if "prdc" in self.choices:
            self._prdc.update(data, real=True)
        if "sfid" in self.choices:
            self._sfid.update(data, real=True)
        if "fdd" in self.choices:
            self._fdd.update(data, real=True)
        if "fvd" in self.choices:
            assert isinstance(data, torch.Tensor) and data.dtype == torch.uint8
            # data is a torch.Tensor of type uint8
            # data = (rearrange(data, "b t c h w -> b t h w c") / 255.0 - 0.5) * 2
            b, t, c, h, w = data.shape
            data = rearrange(data, "b t c h w -> (b t) c h w").float()
            data = F.interpolate(
                data, size=(224, 224), mode="bilinear", align_corners=False
            )
            data = rearrange(data, "(b t) c h w -> b t h w c", t=t).float()
            self._fvd.update(data, real=False)

    def update_fake(self, data):
        if "fid" in self.choices:
            self._fid.update(data, real=False)
        if "kid" in self.choices:
            self._kid.update(data, real=False)
        if "prdc" in self.choices:
            self._prdc.update(data, real=False)
        if "sfid" in self.choices:
            self._sfid.update(data, real=False)
        if "fdd" in self.choices:
            self._fdd.update(data, real=False)
        if "fvd" in self.choices:
            assert isinstance(data, torch.Tensor) and data.dtype == torch.uint8
            # data is a torch.Tensor of type uint8
            # data = (rearrange(data, "b t c h w -> b t h w c") / 255.0 - 0.5) * 2
            b, t, c, h, w = data.shape
            data = rearrange(data, "b t c h w -> (b t) c h w").float()
            data = F.interpolate(
                data, size=(224, 224), mode="bilinear", align_corners=False
            )
            data = rearrange(data, "(b t) c h w -> b t h w c", t=t).float()
            self._fvd.update(data, real=False)

    def compute(self):
        print("computing torchmetrics...")
        _result = dict()
        if "fid" in self.choices:
            fid = self._fid.compute().item()
            _result["num_real"] = self._fid.real_features_num_samples
            _result["num_fake"] = self._fid.fake_features_num_samples
            _result["fid"] = fid
        if "is" in self.choices:
            _is_mean, _is_std = self._is.compute()
            _result["is"] = _is_mean.item()
        if "kid" in self.choices:
            _kid_mean, _kid_std = self._kid.compute()
            _result["kid_mean"] = _kid_mean.item()
            _result["kid_std"] = _kid_std.item()
        if "prdc" in self.choices:
            _prdc_result = self._prdc.compute()
            _prdc_result = {f"prdc_{k}": v for k, v in _prdc_result.items()}
            _result.update(_prdc_result)
        if "sfid" in self.choices:
            sfid = self._sfid.compute().item()
            _result["sfid"] = sfid
        if "fdd" in self.choices:
            fdd = self._fdd.compute().item()
            _result["fdd"] = fdd
        if "fvd" in self.choices:
            fvd = self._fvd.compute().item()
            _result["fvd"] = fvd
        return _result

    def reset(self):
        if "fid" in self.choices:
            self._fid.reset()
        if "is" in self.choices:
            self._is.reset()
        if "kid" in self.choices:
            self._kid.reset()
        if "prdc" in self.choices:
            self._prdc.reset()
        if "sfid" in self.choices:
            self._sfid.reset()
        if "fdd" in self.choices:
            self._fdd.reset()
        if "fvd" in self.choices:
            self._fvd.reset()


if __name__ == "__main__":
    _metric = MyMetric(
        choices=["fid", "is", "kid", "prdc", "sfid", "fdd"],
    )
    _metric.update_real(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )
    _metric.update_fake(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )

    print(_metric.compute())
