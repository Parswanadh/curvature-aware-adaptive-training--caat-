import logging
import yaml
from typing import Callable, Dict, Any

# Helper Functions
def check_nan(tensor):
    """Check if a tensor contains NaN or inf values."""
    return (tensor != tensor).any() or not (-float('inf') < tensor.min() and tensor.max() < float('inf'))

def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

# Configuration Management
def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], file_path: str):
    """Save configuration to a YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Common Operations
def apply_function(tensor, func: Callable[[Any], Any]):
    """Apply a function to each element of the tensor."""
    if check_nan(tensor):
        logger = logging.getLogger(__name__)  # Ensure logger is set correctly
        logger.warning("Tensor contains NaN or inf values, skipping transformation.")
    else:
        return func(tensor)

# Example Usage
if __name__ == "__main__":
    logger = setup_logging()
    config = load_config('config.yaml')
    save_config({'key': 'value'}, 'new_config.yaml')