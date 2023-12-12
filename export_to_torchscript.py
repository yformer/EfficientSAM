import torch
from efficient_sam.efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

torch.jit.save(torch.jit.script(build_efficient_sam_vitt()), "torchscripted_model/efficient_sam_vitt_torchscript.pt")
# torch.jit.save(torch.jit.script(build_efficient_sam_vits()), "torchscripted_model/efficient_sam_vits_torchscript.pt")
