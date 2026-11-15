from nightrad.raw.core import RunConfig
from nightrad.raw.net import NetConfig


def tiny_cfg(**kw):
    net = kw.pop("net", None) or NetConfig(width=8, latent=4)
    kw.setdefault("daynight", "both")  # the real default is night-first; tests use synthetic tiles of both kinds
    return RunConfig(net=net, **kw)
