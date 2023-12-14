import torch
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam
import zipfile
import os

# Efficient SAM (VIT-tiny)
torch.jit.save(torch.jit.script(build_efficient_sam_vitt()), "torchscripted_model/efficient_sam_vitt_torchscript.pt")

# Efficient SAM (VIT-small)
# Since VIT-small is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
torch.jit.save(torch.jit.script(build_efficient_sam_vits()), "torchscripted_model/efficient_sam_vits_torchscript.pt")

# Squeeze SAM (UNET)
# torch.jit.save(torch.jit.script(build_squeeze_sam()), "torchscripted_model/squeeze_sam_torchscript.pt")
