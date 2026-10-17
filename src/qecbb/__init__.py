__version__ = 'local-refactor-0.1'

# Import perezoso de submodulos (PEP 562): evita arrastrar dependencias pesadas
# u opcionales (p. ej. ``bposd`` via ``bbcodes``/``baselines``) cuando solo se
# necesita un submodulo ligero como ``stim_bb_circuit_data`` o ``model``.
_LAZY = (
    "config", "artifacts", "bbcodes", "graphs", "data", "lif", "model",
    "training", "baselines", "evaluation", "bposd_ref", "fpga",
    "stim_bb", "stim_bb_faithful", "stim_bb_circuit_data", "stim_data",
    "stim_utils",
)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
