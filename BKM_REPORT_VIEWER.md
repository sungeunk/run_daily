http://dg2raptorlake.ikor.intel.com:8501/

## Settings
dg2raptorlake
src: /home/sungeunk/repo/run_daily/scripts/run_daily_report_viewer.py
service file: /etc/systemd/system/viewer_daily_report.service
```bash
[Unit]
 Description=Daily report viewer

[Service]
 User=sungeunk
 WorkingDirectory=/home/sungeunk/repo/run_daily/scripts
 ExecStart=/home/sungeunk/miniforge3/envs/daily/bin/python -m streamlit run /home/sungeunk/repo/run_daily/scripts/run_daily_report_viewer.py
 Restart=always

[Install]
 WantedBy=multi-user.target
```


## Start service
* service name is the filename.
sudo systemctl daemon-reload
sudo systemctl stop viewer_daily_report
sudo systemctl start viewer_daily_report
sudo systemctl status viewer_daily_report

sudo systemctl restart viewer_daily_report

