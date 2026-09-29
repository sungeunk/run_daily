#!/usr/bin/env bash
set -Eeuo pipefail

rsync -avzhP \
	sungeunk@dg2fizz.ikor.intel.com:/mnt/hdd/model/ov-share-13.sclab.intel.com/cv_bench_cache/ \
	/c/dev/models/daily/
