#!/bin/bash

. $1

# python -m pytest daily/tests/ -vv --device=GPU.1 --short-run
python daily/run.py --device GPU.1 --short-run
