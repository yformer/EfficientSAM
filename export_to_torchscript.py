import torch
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import zipfile
import os

torch.jit.save(torch.jit.script(build_efficient_sam_vitt()), "torchscripted_model/efficient_sam_vitt_torchscript.pt")

# Since VIT-small is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
torch.jit.save(torch.jit.script(build_efficient_sam_vits()), "torchscripted_model/efficient_sam_vits_torchscript.pt")
