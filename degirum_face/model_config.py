#
# model_config.py: Model configuration loader and management system
#
# Copyright DeGirum Corporation 2025
#

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict


class ModelConfig:
    """Model configuration loader that derives all information from model entries."""

    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            config_file = Path(__file__).parent / "models.yaml"

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        # Validate configuration before building tables
        validation_issues = self._validate_before_build()
        if validation_issues:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"  • {issue}" for issue in validation_issues)
            )

        # Build lookup tables on initialization
        self._build_lookup_tables()

    def _validate_before_build(self) -> List[str]:
        """Validate configuration before building lookup tables."""
        issues = []
        models = self.config.get("models", {})

        # Check for required fields
        for model_name, model_info in models.items():
            if not model_info.get("task"):
                issues.append(f"Model '{model_name}' missing required 'task' field")
            if not model_info.get("hardware"):
                issues.append(f"Model '{model_name}' missing required 'hardware' field")
            if not model_info.get("description"):
                issues.append(f"Model '{model_name}' missing 'description' field")

        # Check for multiple defaults for same hardware/task (CRITICAL CHECK)
        defaults_check = defaultdict(list)
        for model_name, model_info in models.items():
            if model_info.get("is_default", False):
                key = (model_info.get("hardware"), model_info.get("task"))
                defaults_check[key].append(model_name)

        for (hw, task), models_list in defaults_check.items():
            if len(models_list) > 1:
                issues.append(
                    f"CRITICAL: Multiple default models for {hw}/{task}: {models_list}"
                )

        return issues

    def _build_lookup_tables(self):
        """Build all lookup tables from model entries."""
        models = self.config.get("models", {})

        # Initialize collections
        self._all_tasks = set()
        self._all_hardware = set()
        self._hardware_tasks = defaultdict(set)
        self._task_hardware = defaultdict(set)
        self._task_models = defaultdict(list)
        self._hardware_models = defaultdict(list)
        self._hardware_task_models = defaultdict(lambda: defaultdict(list))
        self._hardware_zoo_urls = {}
        self._defaults = defaultdict(dict)

        # Build tables from models
        for model_name, model_info in models.items():
            task = model_info.get("task")
            hardware = model_info.get("hardware")
            zoo_url = model_info.get("zoo_url", "degirum/public")
            is_default = model_info.get("is_default", False)

            if task:
                self._all_tasks.add(task)
                self._task_models[task].append(model_name)

            if hardware:
                self._all_hardware.add(hardware)
                self._hardware_models[hardware].append(model_name)
                self._hardware_zoo_urls[hardware] = zoo_url

            if task and hardware:
                self._hardware_tasks[hardware].add(task)
                self._task_hardware[task].add(hardware)
                self._hardware_task_models[hardware][task].append(model_name)

                if is_default:
                    self._defaults[hardware][task] = model_name

        # Convert sets to sorted lists for consistency
        self._all_tasks = sorted(self._all_tasks)
        self._all_hardware = sorted(self._all_hardware)
        for hw in self._hardware_tasks:
            self._hardware_tasks[hw] = sorted(self._hardware_tasks[hw])
        for task in self._task_hardware:
            self._task_hardware[task] = sorted(self._task_hardware[task])

    # ================== BASIC QUERIES ==================

    def get_all_tasks(self) -> List[str]:
        """Get list of all supported tasks (derived from models)."""
        return self._all_tasks

    def get_all_hardware(self) -> List[str]:
        """Get list of all supported hardware devices (derived from models)."""
        return self._all_hardware

    def get_all_models(self) -> List[str]:
        """Get list of all available models."""
        return list(self.config.get("models", {}).keys())

    def get_inference_hosts(self) -> List[str]:
        """Get list of available inference host addresses."""
        return list(self.config.get("inference_hosts", {}).keys())

    # ================== HARDWARE QUERIES ==================

    def get_tasks_for_hardware(self, hardware: str) -> List[str]:
        """Get list of supported tasks for a given hardware device (derived from models)."""
        return list(self._hardware_tasks.get(hardware, []))

    def get_hardware_for_task(self, task: str) -> List[str]:
        """Get list of hardware devices that support a given task (derived from models)."""
        return list(self._task_hardware.get(task, []))

    def get_zoo_url(self, hardware: str) -> str:
        """Get zoo URL for a hardware device (from any model using that hardware)."""
        return self._hardware_zoo_urls.get(hardware, "degirum/public")

    def get_hardware_description(self, hardware: str) -> str:
        """Get description for a hardware device."""
        hw_defs = self.config.get("hardware_definitions", {})
        return hw_defs.get(hardware, {}).get("description", f"{hardware} device")

    # ================== MODEL QUERIES ==================

    def get_models_for_task(self, task: str) -> List[str]:
        """Get all models that support a given task (derived from models)."""
        return self._task_models.get(task, [])

    def get_models_for_hardware(self, hardware: str) -> List[str]:
        """Get all models available for a given hardware device (derived from models)."""
        return self._hardware_models.get(hardware, [])

    def get_models_for_task_and_hardware(self, task: str, hardware: str) -> List[str]:
        """Get models for a specific task on a specific hardware device (derived from models)."""
        return self._hardware_task_models.get(hardware, {}).get(task, [])

    def get_tasks_for_model(self, model_name: str) -> List[str]:
        """Get tasks that a given model supports."""
        model_info = self.config.get("models", {}).get(model_name, {})
        task = model_info.get("task")
        return [task] if task else []

    def get_hardware_for_model(self, model_name: str) -> List[str]:
        """Get hardware devices that support a given model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        hardware = model_info.get("hardware")
        return [hardware] if hardware else []

    # ================== MODEL METADATA QUERIES ==================

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get full metadata for a model."""
        return self.config.get("models", {}).get(model_name, {})

    def get_model_description(self, model_name: str) -> str:
        """Get description for a model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("description", "")

    def get_model_performance(self, model_name: str) -> str:
        """Get performance rating for a model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("performance", "")

    def get_model_accuracy(self, model_name: str) -> str:
        """Get accuracy rating for a model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("accuracy", "")

    def get_model_input_size(self, model_name: str) -> List[int]:
        """Get input size for a model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("input_size", [])

    def get_model_zoo_url(self, model_name: str) -> str:
        """Get zoo URL for a specific model."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("zoo_url", "degirum/public")

    # ================== DEFAULT MODEL QUERIES ==================

    def get_default_model(self, hardware: str, task: str) -> str:
        """Get default model for a hardware device and task (derived from models)."""
        return self._defaults.get(hardware, {}).get(task, "")

    def get_all_defaults(self) -> Dict[str, Dict[str, str]]:
        """Get all default model mappings (derived from models)."""
        return dict(self._defaults)

    # ================== FILTERING AND SEARCH ==================

    def filter_models_by_performance(self, performance: str) -> List[str]:
        """Get models filtered by performance rating."""
        models = []
        for model_name, model_info in self.config.get("models", {}).items():
            if model_info.get("performance") == performance:
                models.append(model_name)
        return models

    def filter_models_by_accuracy(self, accuracy: str) -> List[str]:
        """Get models filtered by accuracy rating."""
        models = []
        for model_name, model_info in self.config.get("models", {}).items():
            if model_info.get("accuracy") == accuracy:
                models.append(model_name)
        return models

    def search_models(
        self,
        task: str = None,
        hardware: str = None,
        performance: str = None,
        accuracy: str = None,
    ) -> List[str]:
        """Search models with multiple filters."""
        models = []
        for model_name, model_info in self.config.get("models", {}).items():
            # Apply filters
            if task and model_info.get("task") != task:
                continue
            if hardware and model_info.get("hardware") != hardware:
                continue
            if performance and model_info.get("performance") != performance:
                continue
            if accuracy and model_info.get("accuracy") != accuracy:
                continue

            models.append(model_name)
        return models

    # ================== VALIDATION ==================

    def validate_hardware_task_combination(self, hardware: str, task: str) -> bool:
        """Check if a hardware device supports a given task (derived from models)."""
        return task in self._hardware_tasks.get(hardware, set())

    def validate_model_hardware_combination(
        self, model_name: str, hardware: str
    ) -> bool:
        """Check if a model is available for a given hardware device."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("hardware") == hardware

    def validate_model_task_combination(self, model_name: str, task: str) -> bool:
        """Check if a model supports a given task."""
        model_info = self.config.get("models", {}).get(model_name, {})
        return model_info.get("task") == task

    # ================== TASK METADATA ==================

    def get_task_description(self, task: str) -> str:
        """Get description for a task."""
        task_defs = self.config.get("task_definitions", {})
        return task_defs.get(task, {}).get("description", f"{task} task")

    def get_task_output_format(self, task: str) -> str:
        """Get output format for a task."""
        task_defs = self.config.get("task_definitions", {})
        return task_defs.get(task, {}).get("output_format", "")

    # ================== CONSISTENCY CHECKS ==================

    def validate_configuration(self) -> List[str]:
        """Validate configuration consistency and return list of issues."""
        issues = []
        models = self.config.get("models", {})

        # Check for required fields
        for model_name, model_info in models.items():
            if not model_info.get("task"):
                issues.append(f"Model '{model_name}' missing required 'task' field")
            if not model_info.get("hardware"):
                issues.append(f"Model '{model_name}' missing required 'hardware' field")
            if not model_info.get("description"):
                issues.append(f"Model '{model_name}' missing 'description' field")
            if not model_info.get("zoo_url"):
                issues.append(f"Model '{model_name}' missing 'zoo_url' field")

        # Check for multiple defaults for same hardware/task (THE CRITICAL ISSUE)
        defaults_check = defaultdict(list)
        for model_name, model_info in models.items():
            if model_info.get("is_default", False):
                key = (model_info.get("hardware"), model_info.get("task"))
                defaults_check[key].append(model_name)

        for (hw, task), models_list in defaults_check.items():
            if len(models_list) > 1:
                issues.append(
                    f"CRITICAL: Multiple default models for {hw}/{task}: {models_list}"
                )

        # Check for hardware/task combinations without any default
        for hw in self.get_all_hardware():
            for task in self.get_tasks_for_hardware(hw):
                if not self.get_default_model(hw, task):
                    models_for_combo = self.get_models_for_task_and_hardware(task, hw)
                    if models_for_combo:  # Only warn if there are models but no default
                        issues.append(
                            f"WARNING: No default model for {hw}/{task} (available: {models_for_combo})"
                        )

        return issues

    def get_duplicate_defaults(self) -> Dict[str, List[str]]:
        """Get all hardware/task combinations with multiple default models."""
        duplicates = {}
        models = self.config.get("models", {})

        defaults_check = defaultdict(list)
        for model_name, model_info in models.items():
            if model_info.get("is_default", False):
                key = f"{model_info.get('hardware')}/{model_info.get('task')}"
                defaults_check[key].append(model_name)

        for combo, models_list in defaults_check.items():
            if len(models_list) > 1:
                duplicates[combo] = models_list

        return duplicates

    def suggest_fix_for_duplicates(self) -> List[str]:
        """Suggest fixes for duplicate default models."""
        suggestions = []
        duplicates = self.get_duplicate_defaults()

        for combo, models_list in duplicates.items():
            hw, task = combo.split("/")
            suggestions.append(f"For {combo}:")
            suggestions.append(f"  Choose ONE model to keep as default:")

            for i, model in enumerate(models_list):
                model_info = self.get_model_info(model)
                perf = model_info.get("performance", "unknown")
                acc = model_info.get("accuracy", "unknown")
                suggestions.append(f"    {i+1}. {model} (perf: {perf}, acc: {acc})")

            suggestions.append(f"  Set is_default: false for the others")
            suggestions.append("")

        return suggestions

    # ================== STATISTICS ==================

    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "total_models": len(self.get_all_models()),
            "total_tasks": len(self.get_all_tasks()),
            "total_hardware": len(self.get_all_hardware()),
            "models_per_task": {
                task: len(self.get_models_for_task(task))
                for task in self.get_all_tasks()
            },
            "models_per_hardware": {
                hw: len(self.get_models_for_hardware(hw))
                for hw in self.get_all_hardware()
            },
            "tasks_per_hardware": {
                hw: len(self.get_tasks_for_hardware(hw))
                for hw in self.get_all_hardware()
            },
        }


# Global config instance
_model_config = None


def get_model_config() -> ModelConfig:
    """Get global model configuration instance."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config
