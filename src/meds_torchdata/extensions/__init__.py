_HAS_LIGHTNING = False

try:
    import lightning

    _HAS_LIGHTNING = True
except ImportError:
    pass

if _HAS_LIGHTNING:
    from .lightning import Datamodule
