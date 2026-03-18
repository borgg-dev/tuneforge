"""
Model manager for TuneForge music generation backends.

Provides lazy-loading, backend switching, and GPU monitoring
for multiple music generation model backends.
"""

import time
from typing import Any

import numpy as np
import torch
from loguru import logger


class ModelManager:
    """Manages music generation model backends with lazy loading.

    Supports switching between MusicGen and Stable Audio backends
    at runtime. Only one backend is loaded at a time to conserve GPU memory.
    """

    SUPPORTED_BACKENDS: tuple[str, ...] = ("musicgen", "stable_audio", "diffrhythm", "heartmula")

    def __init__(
        self,
        default_backend: str = "musicgen",
        model_size: str = "medium",
        device: str = "cuda",
    ) -> None:
        """Initialize the model manager.

        Args:
            default_backend: Backend to use by default ("musicgen" or "stable_audio").
            model_size: Model size for MusicGen ("small", "medium", "large").
            device: Torch device string.
        """
        if default_backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{default_backend}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        self._default_backend = default_backend
        self._model_size = model_size
        self._device = device
        self._backends: dict[str, Any] = {}
        self._active_backend_name: str = default_backend
        self._generation_count: int = 0
        self._total_generation_time: float = 0.0

        logger.info(
            f"ModelManager initialized: backend={default_backend}, "
            f"model_size={model_size}, device={device}"
        )

    def preload(self) -> None:
        """Preload the active backend model into memory.

        Called at startup so the first challenge doesn't timeout
        waiting for model download/load.
        """
        backend = self._get_or_create_backend(self._active_backend_name)
        backend.load()
        logger.info(f"Backend '{self._active_backend_name}' preloaded and ready")

    @property
    def active_backend(self) -> str:
        """Name of the currently active backend."""
        return self._active_backend_name

    @property
    def active_model_id(self) -> str:
        """Model identifier of the active backend."""
        backend = self._get_or_create_backend(self._active_backend_name)
        return backend.model_id

    @property
    def supports_vocals(self) -> bool:
        """Whether the active backend supports vocals/lyrics."""
        backend = self._get_or_create_backend(self._active_backend_name)
        return getattr(backend, "SUPPORTS_VOCALS", False)

    @property
    def generation_count(self) -> int:
        """Total number of generations performed."""
        return self._generation_count

    @property
    def average_generation_time_ms(self) -> float:
        """Rolling average generation time in milliseconds."""
        if self._generation_count == 0:
            return 0.0
        return (self._total_generation_time / self._generation_count) * 1000

    def _get_or_create_backend(self, name: str) -> Any:
        """Get an existing backend instance or create a new one.

        Args:
            name: Backend name.

        Returns:
            Backend instance (MusicGenBackend or StableAudioBackend).
        """
        if name in self._backends:
            return self._backends[name]

        if name == "musicgen":
            from tuneforge.generation.musicgen_backend import MusicGenBackend

            backend = MusicGenBackend(
                model_size=self._model_size,
                device=self._device,
            )
        elif name == "stable_audio":
            from tuneforge.generation.stable_audio_backend import StableAudioBackend

            backend = StableAudioBackend(device=self._device)
        elif name == "diffrhythm":
            from tuneforge.generation.diffrhythm_backend import DiffRhythmBackend

            use_full = self._model_size == "full"
            backend = DiffRhythmBackend(device=self._device, use_full_model=use_full)
        elif name == "heartmula":
            from tuneforge.generation.heartmula_backend import HeartMuLaBackend

            backend = HeartMuLaBackend(
                device=self._device,
                version=self._model_size if self._model_size in ("3B", "7B") else "3B",
            )
        else:
            raise ValueError(f"Unknown backend: {name}")

        self._backends[name] = backend
        return backend

    def switch_backend(self, name: str) -> None:
        """Switch to a different generation backend.

        Unloads the current backend model from GPU memory before
        loading the new one.

        Args:
            name: Backend name ("musicgen" or "stable_audio").
        """
        if name not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{name}'. Supported: {self.SUPPORTED_BACKENDS}"
            )

        if name == self._active_backend_name:
            logger.debug(f"Already using backend: {name}")
            return

        # Unload current backend to free GPU memory
        current = self._backends.get(self._active_backend_name)
        if current is not None and current.is_loaded:
            logger.info(f"Unloading backend: {self._active_backend_name}")
            current.unload()

        self._active_backend_name = name
        logger.info(f"Switched to backend: {name}")

    def generate(
        self,
        prompt: str,
        duration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        """Generate audio using the active backend.

        Args:
            prompt: Natural language description of desired music.
            duration: Desired audio duration in seconds.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters passed to the backend.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        backend = self._get_or_create_backend(self._active_backend_name)

        logger.info(
            f"Generating with {self._active_backend_name}: "
            f"duration={duration}s, seed={seed}"
        )
        t0 = time.time()

        try:
            audio, sr = backend.generate(
                prompt=prompt,
                duration_seconds=duration,
                seed=seed,
                **kwargs,
            )
            elapsed = time.time() - t0
            self._generation_count += 1
            self._total_generation_time += elapsed
            logger.info(
                f"Generation #{self._generation_count} complete: "
                f"{elapsed:.1f}s, avg={self.average_generation_time_ms:.0f}ms"
            )
            return audio, sr

        except Exception as exc:
            logger.error(f"Generation failed on {self._active_backend_name}: {exc}")
            raise

    def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU information and memory usage.

        Returns:
            Dictionary with GPU model, memory allocated, memory reserved,
            and total memory in MB.
        """
        info: dict[str, Any] = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_model": "N/A",
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "memory_total_mb": 0.0,
            "utilization_percent": 0.0,
        }

        if not torch.cuda.is_available():
            return info

        try:
            device_idx = 0
            if ":" in self._device:
                try:
                    device_idx = int(self._device.split(":")[1])
                except (ValueError, IndexError):
                    device_idx = 0

            info["gpu_model"] = torch.cuda.get_device_name(device_idx)
            info["memory_allocated_mb"] = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            info["memory_reserved_mb"] = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_idx).total_memory
            info["memory_total_mb"] = total / (1024 ** 2)

            if total > 0:
                allocated = torch.cuda.memory_allocated(device_idx)
                info["utilization_percent"] = (allocated / total) * 100.0

        except Exception as exc:
            logger.warning(f"Failed to get GPU info: {exc}")

        return info

    def unload_all(self) -> None:
        """Unload all backends and free GPU memory."""
        for name, backend in self._backends.items():
            if backend.is_loaded:
                logger.info(f"Unloading backend: {name}")
                backend.unload()
        self._backends.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All backends unloaded")
