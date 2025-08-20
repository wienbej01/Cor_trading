"""
Configuration management module for FX-Commodity correlation arbitrage strategy.
Handles loading and validation of YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigManager:
    """Manages loading and access to configuration files."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration directory. If None, defaults to ../configs.
        """
        if config_path is None:
            # Default to ../configs relative to this file
            self.config_path = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_path = config_path
            
        self._pairs_config: Optional[Dict[str, Any]] = None
        
    def load_pairs_config(self) -> Dict[str, Any]:
        """
        Load the pairs configuration from YAML file.
        
        Returns:
            Dictionary containing pairs configuration.
            
        Raises:
            FileNotFoundError: If pairs.yaml is not found.
            yaml.YAMLError: If YAML parsing fails.
        """
        if self._pairs_config is None:
            config_file = self.config_path / "pairs.yaml"
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
                
            logger.info(f"Loading pairs configuration from {config_file}")
            
            with open(config_file, "r") as f:
                self._pairs_config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration for {len(self._pairs_config)} pairs")
            
        return self._pairs_config
        
    def get_pair_config(self, pair_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pair.
        
        Args:
            pair_name: Name of the pair (e.g., "usdcad_wti").
            
        Returns:
            Dictionary containing configuration for the specified pair.
            
        Raises:
            KeyError: If pair_name is not found in configuration.
        """
        pairs_config = self.load_pairs_config()
        
        if pair_name not in pairs_config:
            raise KeyError(f"Pair '{pair_name}' not found in configuration")
            
        return pairs_config[pair_name]
        
    def list_pairs(self) -> list:
        """
        Get list of available pair names.
        
        Returns:
            List of pair names.
        """
        pairs_config = self.load_pairs_config()
        return list(pairs_config.keys())


# Global instance for easy access
config = ConfigManager()