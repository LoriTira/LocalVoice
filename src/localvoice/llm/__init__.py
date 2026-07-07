from localvoice.config import ConfigError, LlmConfig


def build_llm_engine(cfg: LlmConfig):
    """Construct the engine `cfg.engine` selects: "mlx_lm" (text-only) or
    "mlx_vlm" (text + vision). The single seam every construction site
    (engineset factories, terminal, bench) should go through, so a config
    change to llm.engine is honored identically everywhere.

    MlxLmEngine/MlxVlmEngine are imported here, not at module top — same
    lazy-import discipline as the mlx_lm/mlx_vlm calls inside those classes'
    own load()/stream(): nothing MLX-related should load just because this
    module (or localvoice.llm) was imported.
    """
    if cfg.engine == "mlx_lm":
        from localvoice.llm.mlx_lm_engine import MlxLmEngine

        return MlxLmEngine(cfg)
    if cfg.engine == "mlx_vlm":
        from localvoice.llm.mlx_vlm_engine import MlxVlmEngine

        return MlxVlmEngine(cfg)
    raise ConfigError(f"unknown llm.engine: {cfg.engine!r}")
