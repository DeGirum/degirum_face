#
# base_model_component.py: Shared base class for model components (detector, embedder)
#
# Copyright DeGirum Corporation 2025
#

import logging
from typing import Any, Optional, Tuple, Dict, List, Union
from .model_config import get_model_config
from .pipeline_config import ModelSpec

logger = logging.getLogger(__name__)


class BaseModelComponent:

    @classmethod
    def get_supported_hardware(cls) -> List[str]:
        """
        Get list of hardware devices that support this model component's task.
        """
        return cls.get_supported_hardware_for_task(cls.TASK)

    @classmethod
    def get_available_models(
        cls, hardware: Optional[str] = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get list of available models for this model component's task.
        """
        return cls.get_available_models_for_task(cls.TASK, hardware)

    @staticmethod
    def get_supported_hardware_for_task(task: str) -> List[str]:
        config = get_model_config()
        return config.get_hardware_for_task(task)

    @staticmethod
    def get_available_models_for_task(
        task: str, hardware: Optional[str] = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        config = get_model_config()
        if hardware is not None:
            return config.get_models_for_task_and_hardware(task, hardware)
        else:
            all_hardware = config.get_hardware_for_task(task)
            result = {}
            for hw in all_hardware:
                models = config.get_models_for_task_and_hardware(task, hw)
                if models:
                    result[hw] = models
            return result

    TASK = None  # To be set by subclass

    @classmethod
    def _load_auto_mode_model(
        cls, hardware: str, inference_host_address: str = "@cloud"
    ) -> Tuple[Any, str, str]:
        config = get_model_config()
        if not config.validate_hardware_task_combination(hardware, cls.TASK):
            available_tasks = config.get_tasks_for_hardware(hardware)
            raise ValueError(
                f"Hardware '{hardware}' does not support task '{cls.TASK}'. "
                f"Available tasks: {available_tasks}"
            )
        model_name = config.get_default_model(hardware, cls.TASK)
        if not model_name:
            available_models = config.get_models_for_task_and_hardware(
                cls.TASK, hardware
            )
            if available_models:
                model_name = available_models[0]
                logger.warning(
                    f"No default model for {hardware}/{cls.TASK}, using {model_name}"
                )
            else:
                raise ValueError(
                    f"No models available for {hardware}/{cls.TASK}. "
                    f"Available hardware: {config.get_hardware_for_task(cls.TASK)}"
                )
        zoo_url = config.get_model_zoo_url(model_name)
        try:
            import degirum as dg

            zoo = dg.connect(inference_host_address, zoo_url)
            model = zoo.load_model(model_name)
            logger.info(
                f"Auto mode: Loaded '{model_name}' on {hardware} via {inference_host_address}"
            )
            return model, model_name, zoo_url
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' for {hardware}: {e}"
            ) from e

    @classmethod
    def auto(
        cls,
        hardware: str,
        inference_host_address: str = "@cloud",
    ):
        model, model_name, zoo_url = cls._load_auto_mode_model(
            hardware, inference_host_address
        )
        return cls(
            hardware=hardware,
            model=model,
            model_name=model_name,
            zoo_url=zoo_url,
            inference_host_address=inference_host_address,
            _creation_mode="auto_factory",
        )

    @classmethod
    def custom(cls, model_spec: ModelSpec):
        try:
            model = model_spec.load_model()
            logger.info(
                f"Custom mode: Loaded '{model_spec.model_name}' from {model_spec.zoo_url} via {model_spec.inference_host_address}"
            )
            return cls(
                hardware=None,
                model=model,
                model_name=model_spec.model_name,
                zoo_url=model_spec.zoo_url,
                inference_host_address=model_spec.inference_host_address,
                _creation_mode="custom",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load custom model '{model_spec.model_name}' from {model_spec.zoo_url}: {e}"
            ) from e

    @classmethod
    def from_config(
        cls,
        model_name: str,
        inference_host_address: str = "@cloud",
    ):
        config = get_model_config()
        model_info = config.get_model_info(model_name)
        if not model_info:
            available_models = config.get_models_for_task(cls.TASK)
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {available_models}"
            )
        hardware = model_info.get("hardware")
        task = model_info.get("task")
        zoo_url = model_info.get("zoo_url", "degirum/public")
        if task != cls.TASK:
            raise ValueError(
                f"Model '{model_name}' is for task '{task}', not '{cls.TASK}'. "
                f"Use {cls.__name__} only for {cls.TASK} models."
            )
        try:
            import degirum as dg

            zoo = dg.connect(inference_host_address, zoo_url)
            model = zoo.load_model(model_name)
            logger.info(
                f"Config mode: Loaded '{model_name}' (hardware: {hardware}) via {inference_host_address}"
            )
            return cls(
                hardware=hardware,
                model=model,
                model_name=model_name,
                zoo_url=zoo_url,
                inference_host_address=inference_host_address,
                _creation_mode="from_config",
            )
        except ConnectionError as e:
            raise RuntimeError(f"Failed to connect to zoo at '{zoo_url}': {e}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                raise RuntimeError(
                    f"Model '{model_name}' not found in zoo '{zoo_url}': {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' from {zoo_url}: {e}"
                ) from e

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hardware": self.hardware,
            "inference_host": self.inference_host_address,
            "zoo_url": self.zoo_url,
            "task": self.TASK,
            "creation_mode": self._creation_mode,
        }

    @staticmethod
    def get_supported_hardware(task: str) -> List[str]:
        config = get_model_config()
        return config.get_hardware_for_task(task)

    @staticmethod
    def get_available_models(
        task: str, hardware: Optional[str] = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        config = get_model_config()
        if hardware is not None:
            return config.get_models_for_task_and_hardware(task, hardware)
        else:
            all_hardware = config.get_hardware_for_task(task)
            result = {}
            for hw in all_hardware:
                models = config.get_models_for_task_and_hardware(task, hw)
                if models:
                    result[hw] = models
            return result
