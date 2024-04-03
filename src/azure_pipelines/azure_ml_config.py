from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class AzureMlConfig:
    subscription_id: str
    resource_group: str
    workspace_name: str

    @classmethod
    def load_config(cls, config_path: Path) -> "AzureMlConfig":
        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)
        return cls(**config_data)