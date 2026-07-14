#!/bin/bash

. $1

python scripts/run_llm_daily.py --device GPU.1 --test_filter TestBenchmark -m /var/www/html/models/daily
