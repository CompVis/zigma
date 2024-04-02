__version__ = "1.0.1"

from dis_mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
from dis_mamba.mamba_ssm.modules.mamba_simple import Mamba
from dis_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
