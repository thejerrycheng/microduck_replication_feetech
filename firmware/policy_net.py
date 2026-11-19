"""The duck's policies as plain arithmetic: numpy only, no ONNX runtime. Shared by the Mac (policy_pack.py checks it
against the ONNX exports) and the duck (which has numpy and nothing else to run a network with).

Every export is the same actor: 61 observations, normalised as (obs - mean) / scale, then three ELU layers of 512, 256
and 128 units and a linear layer to 14 actions. `Net(arrays)` holds one; `load(path)` gives {role: Net} from the pack
written by policy_pack.py."""
import numpy as np


class Net:
    def __init__(self, a):
        f = lambda k: np.asarray(a[k], dtype=np.float32)
        self.mean, self.scale = f("mean"), f("scale")
        self.w = [f(f"w{i}") for i in range(4)]
        self.b = [f(f"b{i}") for i in range(4)]

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float32).reshape(-1) - self.mean) / self.scale
        for i in range(3):
            x = self.w[i] @ x + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))        # ELU
        return self.w[3] @ x + self.b[3]


def load(path):
    z = np.load(path)
    roles = sorted({k.split("/")[0] for k in z.files})
    return {r: Net({k.split("/")[1]: z[k] for k in z.files if k.startswith(r + "/")}) for r in roles}
