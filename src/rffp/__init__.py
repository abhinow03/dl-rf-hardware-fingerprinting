"""rffp — deep-learning RF hardware fingerprinting & open-world emitter discovery.

Subpackages:
    models      dual-branch RFEncoder + SupCon loss + generalized encoder
    data        dataset loaders / registry (WiSig, DRFF, ORACLE, BLE), STFT front-end
    physics     classical hand-crafted RF features (LPC residual, spectral stats)
    training    training entrypoints (ORACLE closed-set, WiSig metric, LOPO, DANN, BLE)
    evaluation  scoring harness, LOPO benchmark, ORACLE eval, inference, verify gates
    discovery   open-world discovery: protocol router demo, WiSig probes, BLE study
    config      single overridable place for all dataset / checkpoint / output paths
"""
__version__ = "1.0.0"
