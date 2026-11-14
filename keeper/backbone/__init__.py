# Lazy imports to avoid scipy GLIBCXX issues on some systems
# from .MixObviousSubtle_3DGS import MixObviousSubtle3DGS
# from .defense import UnifiedDefense
# from .defense.core import unified_defense_aggregate

def __getattr__(name):
    if name == "MixObviousSubtle3DGS":
        from .MixObviousSubtle_3DGS import MixObviousSubtle3DGS
        return MixObviousSubtle3DGS
    elif name == "UnifiedDefense":
        from .defense import UnifiedDefense
        return UnifiedDefense
    elif name == "unified_defense_aggregate":
        from .defense.core import unified_defense_aggregate
        return unified_defense_aggregate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

