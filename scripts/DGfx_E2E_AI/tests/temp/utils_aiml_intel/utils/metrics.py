import datetime
import json
import os
import pprint
from typing import Optional

import pandas as pd


class BaseObject:
    def __init__(self):
        self.customized = {}

    def to_dict(self):
        default_values = self.__dict__.copy()
        default_values.pop("customized", None)
        default_values.update(self.customized)

        for k, v in default_values.items():
            if isinstance(v, BaseObject):
                default_values[k] = v.to_dict()

        return {k: v for k, v in default_values.items() if v}


class ModelInfo(BaseObject):
    def __init__(
        self,
        full_name: Optional[str] = None,
        is_huggingface: Optional[bool] = False,
        is_text_generation: Optional[bool] = False,
        short_name: Optional[str] = None,
    ):
        super().__init__()
        self.full_name = full_name
        self.is_huggingface = is_huggingface
        self.is_text_generation = is_text_generation
        self.short_name = short_name
        self.input_shape = []


class BackendOptions(BaseObject):
    def __init__(
        self,
        enable_profiling: Optional[bool] = False,
        execution_provider: Optional[str] = None,
        use_io_binding: Optional[bool] = False,
    ):
        super().__init__()
        self.enable_profiling = enable_profiling
        self.execution_provider = execution_provider
        self.use_io_binding = use_io_binding


class Config(BaseObject):
    def __init__(
        self,
        backend: Optional[str] = "onnxruntime-genai",
        batch_size: Optional[int] = 1,
        seq_length: Optional[int] = 0,
        precision: Optional[str] = "fp32",
        warmup_runs: Optional[int] = 1,
        measured_runs: Optional[int] = 10,
    ):
        super().__init__()
        self.backend = backend
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.precision = precision
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.model_info = ModelInfo()
        self.backend_options = BackendOptions()


class Metadata(BaseObject):
    def __init__(
        self,
        device: Optional[str] = None,
        package_name: Optional[str] = None,
        package_version: Optional[str] = None,
        platform: Optional[str] = None,
        python_version: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.package_name = package_name
        self.package_version = package_version
        self.platform = platform
        self.python_version = python_version


class Metrics(BaseObject):
    def __init__(
        self,
        latency_ms_mean: Optional[float] = 0.0,
        throughput_qps: Optional[float] = 0.0,
        max_memory_usage_GB: Optional[float] = 0.0,
    ):
        super().__init__()
        self.latency_ms_mean = latency_ms_mean
        self.throughput_qps = throughput_qps
        self.max_memory_usage_GB = max_memory_usage_GB
        
class Logging(BaseObject):
    def __init__(
        self,
    ):
        super().__init__()


class BenchmarkRecord:
    def __init__(
        self,
        model_name: str,
        precision: str,
        backend: str,
        device: str,
        package_name: str,
        package_version: str,
        trigger_date: Optional[str] = None,
    ):
        self.config = Config()
        self.metrics = Metrics()
        self.logging = Logging()
        self.metadata = Metadata()
        self.trigger_date = trigger_date or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.config.model_info.full_name = model_name
        self.config.precision = precision
        self.config.backend = backend
        self.metadata.device = device
        self.metadata.package_name = package_name
        self.metadata.package_version = package_version

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "logging": self.logging.to_dict(),
            "trigger_date": self.trigger_date,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def save_as_csv(cls, file_name: str, records: list) -> None:
        file_name = file_name.with_suffix(".csv")
        if records is None or len(records) == 0:
            return
        rds = [record.to_dict() for record in records]
        df = pd.json_normalize(rds)
        df.to_csv(file_name, index=False)
        print(f"Results saved in {file_name}!")

    @classmethod
    def save_as_json(cls, file_name: str, records: list) -> None:
        file_name = file_name.with_suffix(".json")
        if records is None or len(records) == 0:
            return
        rds = [record.to_dict() for record in records]
        with open(file_name, "w") as f:
            json.dump(rds, f, indent=4, default=str)
        print(f"Results saved in {file_name}!")
            
    @classmethod
    def save_as_txt(cls, file_name: str, records: list) -> None:
        file_name = file_name.with_suffix(".txt")
        if records is None or len(records) == 0:
            return
        try:
            rds = pprint.pformat([record.to_dict() for record in records][0])
        except AttributeError: # for payload that's already in dic format
            rds = pprint.pformat(records[0])
        with open(file_name, "w", encoding='utf-8') as f:
            f.write(rds)
        print(f"Results saved in {file_name}!")
                
    @classmethod
    def print_for_test_e2e_plugin(cls, records: list) -> None:
        if records is None or len(records) == 0:
            return
        rds = [record.to_dict() for record in records]
        print(f"Overall Score: {rds[0]['logging']}")
        
    def import_accuracy_score(self, path) -> None:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            accuracy = df.values.tolist()[0][-1]
            self.metrics.customized["Similarity Score"] = round(accuracy, 2)
        else:
            print(f"Path: {path} does not contain accuracy score")
            return