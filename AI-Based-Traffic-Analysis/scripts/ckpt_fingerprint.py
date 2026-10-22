import hashlib
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu")
h = hashlib.sha256()
for k in sorted(ck["state"]):
    h.update(k.encode())
    h.update(ck["state"][k].detach().cpu().numpy().tobytes())
print(h.hexdigest()[:16], "arch=%s window=%s horizon=%s" % (ck["arch"], ck["window"], ck["horizon"]))
print({k: v for k, v in ck.items() if k not in ("state", "norm")})