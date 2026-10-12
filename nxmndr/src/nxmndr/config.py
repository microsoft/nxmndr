# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Configuration management for the inference system.

This module provides centralized configuration using environment variables
and structured configuration with validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class InferenceConfig:
    """Main configuration for the inference system."""

    # General settings
    log_level: str = field(default_factory=lambda: os.getenv("INFERENCE_LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("INFERENCE_LOG_FORMAT", "human"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("INFERENCE_LOG_FILE"))

    # Performance settings
    max_batch_size: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_MAX_BATCH_SIZE", "32"))
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "30.0"))
    )
    enable_memory_pooling: bool = field(
        default_factory=lambda: os.getenv("INFERENCE_ENABLE_MEMORY_POOLING", "true").lower()
        == "true"
    )
    memory_pool_size_mb: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_MEMORY_POOL_SIZE_MB", "1024"))
    )

    # Model cache settings
    model_cache_size: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_MODEL_CACHE_SIZE", "10"))
    )
    model_cache_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("INFERENCE_MODEL_CACHE_DIR")
    )
    preprocess_cache_size: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_PREPROCESS_CACHE_SIZE", "1000"))
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")

        if self.log_format not in ["human", "json"]:
            raise ValueError(f"Invalid log format: {self.log_format}")

        if self.max_batch_size <= 0:
            raise ValueError(f"Max batch size must be positive: {self.max_batch_size}")

        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout_seconds}")


@dataclass
class ServerConfig:
    """Configuration for the gRPC server."""

    # Server settings
    host: str = field(default_factory=lambda: os.getenv("INFERENCE_SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("INFERENCE_SERVER_PORT", "50051")))
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_SERVER_MAX_WORKERS", "4"))
    )
    max_message_size: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_SERVER_MAX_MESSAGE_SIZE", "4194304"))
    )  # 4MB

    # Health check settings
    health_check_interval: float = field(
        default_factory=lambda: float(os.getenv("INFERENCE_HEALTH_CHECK_INTERVAL", "30.0"))
    )

    def __post_init__(self):
        """Validate server configuration."""
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.max_workers <= 0:
            raise ValueError(f"Max workers must be positive: {self.max_workers}")


@dataclass
class RPCConfig:
    """Configuration for Torch RPC."""

    # RPC settings
    backend: str = field(default_factory=lambda: os.getenv("INFERENCE_RPC_BACKEND", "gloo"))
    master_addr: str = field(
        default_factory=lambda: os.getenv("INFERENCE_RPC_MASTER_ADDR", "127.0.0.1")
    )
    master_port: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_RPC_MASTER_PORT", "29500"))
    )
    world_size: int = field(default_factory=lambda: int(os.getenv("INFERENCE_RPC_WORLD_SIZE", "2")))
    worker_timeout: float = field(
        default_factory=lambda: float(os.getenv("INFERENCE_RPC_WORKER_TIMEOUT", "60.0"))
    )

    def __post_init__(self):
        """Validate RPC configuration."""
        if self.backend not in ["gloo", "nccl", "mpi"]:
            raise ValueError(f"Invalid RPC backend: {self.backend}")

        if not 1024 <= self.master_port <= 65535:
            raise ValueError(f"Invalid master port: {self.master_port}")

        if self.world_size < 2:
            raise ValueError(f"World size must be at least 2: {self.world_size}")


@dataclass
class Config:
    """Combined configuration for the entire system."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    rpc: RPCConfig = field(default_factory=RPCConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        """Create configuration from a file (future enhancement)."""
        # TODO: Implement YAML/TOML config file support
        raise NotImplementedError("File-based configuration not yet implemented")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for section_name, section in [
            ("inference", self.inference),
            ("server", self.server),
            ("rpc", self.rpc),
        ]:
            result[section_name] = {
                field.name: getattr(section, field.name)
                for field in section.__dataclass_fields__.values()
            }
        return result


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global _config
    _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration (useful for testing)."""
    global _config
    _config = config


# Environment variable documentation
ENV_VARS_DOC = """
Inference Configuration Environment Variables:

General:
  INFERENCE_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) [default: INFO]
  INFERENCE_LOG_FORMAT: Log format (human, json) [default: human]  
  INFERENCE_LOG_FILE: Optional log file path

Performance:
  INFERENCE_MAX_BATCH_SIZE: Maximum batch size [default: 32]
  INFERENCE_TIMEOUT_SECONDS: Operation timeout [default: 30.0]
  INFERENCE_ENABLE_MEMORY_POOLING: Enable memory pooling [default: true]
  INFERENCE_MEMORY_POOL_SIZE_MB: Memory pool size in MB [default: 1024]

Caching:
  INFERENCE_MODEL_CACHE_SIZE: Number of models to cache [default: 10]
  INFERENCE_MODEL_CACHE_DIR: Directory for cached models
  INFERENCE_PREPROCESS_CACHE_SIZE: Preprocessing cache size [default: 1000]

Server:
  INFERENCE_SERVER_HOST: Server bind host [default: 127.0.0.1]
  INFERENCE_SERVER_PORT: Server port [default: 50051]
  INFERENCE_SERVER_MAX_WORKERS: Max worker threads [default: 4]
  INFERENCE_SERVER_MAX_MESSAGE_SIZE: Max message size in bytes [default: 4194304]
  INFERENCE_HEALTH_CHECK_INTERVAL: Health check interval [default: 30.0]

RPC:
  INFERENCE_RPC_BACKEND: RPC backend (gloo, nccl, mpi) [default: gloo]
  INFERENCE_RPC_MASTER_ADDR: Master address [default: 127.0.0.1]
  INFERENCE_RPC_MASTER_PORT: Master port [default: 29500]
  INFERENCE_RPC_WORLD_SIZE: Number of processes [default: 2]
  INFERENCE_RPC_WORKER_TIMEOUT: Worker timeout [default: 60.0]
"""


if __name__ == "__main__":
    # Print environment variables documentation
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Environment Variables Documentation:")
    logger.info(ENV_VARS_DOC)
