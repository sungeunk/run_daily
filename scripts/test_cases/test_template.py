#!/usr/bin/env python3

import abc
import enum
from tabulate import tabulate


################################################
# Key
################################################

""" Json struct
{
    "tuple(model_name:str, model_config:str, class)": [
        {
            "cmd": str,
            "raw_log": str,
            "test_config": {
                "mem_check": bool
                },
            "return_code": int,
            "data_list": [
                {
                    "in_token": int,
                    "out_token": int,
                    "perf": list[float],
                    "generated_text": str,
                },
            ],
            "process_time" = int,
            "peak_cpu_usage_percent" = float,
            "peak_mem_usage_percent" = float,
            "peak_mem_usage_size" = str,
        },
    ]
}
"""
class CmdItemKey:
    cmd = 'cmd'
    return_code = 'return_code'
    raw_log = 'raw_log'
    test_config = 'test_config'
    data_list = 'data_list'
    process_time = 'process_time'
    peak_cpu_usage_percent = 'peak_cpu_usage_percent'
    peak_mem_usage_percent = 'peak_mem_usage_percent'
    peak_mem_usage_size = 'peak_mem_usage_size'

    class TestConfigKey:
        mem_check = 'mem_check'
        batch = 'batch'

    class DataItemKey:
        perf = 'perf'
        in_token = 'in_token'
        out_token = 'out_token'
        perf = 'perf'
        generated_text = 'generated_text'

class ModelName():
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    chatglm3_6b = 'chatglm3-6b'
    glm_4_9b_chat = 'glm-4-9b-chat'
    llama_2_7b_chat_hf = 'llama-2-7b-chat-hf'
    minicpm_1b_sft = 'minicpm-1b-sft'
    mistral_7b = 'mistral-7b-v0.1'
    phi_3_mini_4k_instruct = 'phi-3-mini-4k-instruct'
    gemma_7b_it = 'gemma-7b-it'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen2_7b = 'qwen2-7b'

class ModelConfig():
    UNKNOWN = ''
    MIXED = 'MIXED'
    INT4 = 'INT4'
    INT8 = 'INT8'
    FP16 = 'FP16'
    OV_FP16_4BIT_DEFAULT = 'OV_FP16-4BIT_DEFAULT'
    OV_FP16_INT4_SYM = 'OV_FP16_INT4_SYM'
    OV_FP16_INT4_SYM_CW = 'OV_FP16_INT4_SYM_CW'


################################################
# Template Class
################################################
class TestTemplate(abc.ABC):
    """
    Return:
    {
        "tuple(model_name:str, model_config:ModelConfig)": [
            {
                "cmd": str,
            }
        ]
    }
    """
    @staticmethod
    @abc.abstractmethod
    def get_command_spec(args) -> dict:
        pass

    """
    Return:
    "result": [
        {
            "in_token": int,
            "out_token": int,
            "perf": list[float],
            "generated_text": str
        },
    ]
    """
    @staticmethod
    @abc.abstractmethod
    def parse_output(args, output) -> list[dict]:
        pass

    @staticmethod
    @abc.abstractmethod
    def generate_report(result_root) -> str:
        return ''

    @staticmethod
    @abc.abstractmethod
    def is_class_name(name) -> bool:
        pass
