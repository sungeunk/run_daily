#!/bin/bash
rsync -avzhP --delete sungeunk@dg2raptorlake.ikor.intel.com:/var/www/html/models/daily/ /c/dev/models/daily/
