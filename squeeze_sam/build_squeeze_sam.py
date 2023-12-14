
from .squeeze_sam import build_squeeze_sam as build_squeeze_sam_base

def build_squeeze_sam():
    return build_squeeze_sam_base("weights/squeeze_sam.pt").eval()
