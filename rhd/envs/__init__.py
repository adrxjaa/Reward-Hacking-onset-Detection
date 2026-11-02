from typing import Any, Dict, Optional, Union

from .region_toggle import ENV_NAME as REGION_TOGGLE_NAME, RegionToggleConfig, RegionToggleEnv
from .coin_collector import ENV_NAME as COIN_COLLECTOR_NAME, CoinCollectorConfig, CoinCollectorEnv
from .battery_collector import ENV_NAME as BATTERY_COLLECTOR_NAME, BatteryCollectorConfig, BatteryCollectorEnv
from .continuous_car import ENV_NAME as CONTINUOUS_CAR_NAME, ContinuousCarConfig, ContinuousCarEnv
from .vacuum_cleaner import ENV_NAME as VACUUM_CLEANER_NAME, VacuumCleanerConfig, VacuumCleanerEnv

ENV_NAME = REGION_TOGGLE_NAME

__all__ = [
    "ENV_NAME",
    "REGION_TOGGLE_NAME",
    "RegionToggleConfig",
    "RegionToggleEnv",
    "COIN_COLLECTOR_NAME",
    "CoinCollectorConfig",
    "CoinCollectorEnv",
    "BATTERY_COLLECTOR_NAME",
    "BatteryCollectorConfig",
    "BatteryCollectorEnv",
    "CONTINUOUS_CAR_NAME",
    "ContinuousCarConfig",
    "ContinuousCarEnv",
    "VACUUM_CLEANER_NAME",
    "VacuumCleanerConfig",
    "VacuumCleanerEnv",
    "create_environment",
    "create_env_config",
]


def create_env_config(env_name: str, **kwargs) -> Any:
    """Instantiate appropriate dataclass config based on environment name or kwargs."""
    name = env_name.lower().replace("-", "_")
    if "coin" in name:
        return CoinCollectorConfig(**kwargs)
    elif "battery" in name:
        return BatteryCollectorConfig(**kwargs)
    elif "vacuum" in name or "cleaner" in name:
        return VacuumCleanerConfig(**kwargs)
    elif "car" in name or "driving" in name or "vehicle" in name:
        return ContinuousCarConfig(**kwargs)
    else:
        return RegionToggleConfig(**kwargs)


def create_environment(
    env_config_or_name: Union[str, Any],
    expose_eval_info: bool = False,
    render_mode: Optional[str] = None,
    **kwargs,
) -> Any:
    """Create environment instance from config object or string name."""
    if isinstance(env_config_or_name, str):
        cfg = create_env_config(env_config_or_name, **kwargs)
    else:
        cfg = env_config_or_name

    if isinstance(cfg, CoinCollectorConfig):
        return CoinCollectorEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
    elif isinstance(cfg, BatteryCollectorConfig):
        return BatteryCollectorEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
    elif isinstance(cfg, VacuumCleanerConfig):
        return VacuumCleanerEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
    elif isinstance(cfg, ContinuousCarConfig):
        return ContinuousCarEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
    elif isinstance(cfg, RegionToggleConfig):
        return RegionToggleEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
    else:
        # Fallback inspection by attributes
        if hasattr(cfg, "coin_pos"):
            return CoinCollectorEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
        elif hasattr(cfg, "battery_pos"):
            return BatteryCollectorEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
        elif hasattr(cfg, "initial_dirt_positions"):
            return VacuumCleanerEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
        elif hasattr(cfg, "track_length") or hasattr(cfg, "speed_reward_weight"):
            return ContinuousCarEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
        else:
            return RegionToggleEnv(cfg, expose_eval_info=expose_eval_info, render_mode=render_mode)
