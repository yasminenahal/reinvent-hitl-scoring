import json
import os
from reinvent_hitl_scoring.configs.config import reinvent_hitl_scoring_config


def _is_development_environment() -> bool:
    try:
        is_dev = reinvent_hitl_scoring_config.get("DEVELOPMENT_ENVIRONMENT", False)
        return is_dev
    except:
        return False
