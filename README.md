# run_daily
Daily test scripts. 

# Initialize environments
- Run init_env.bat first. It will set and install all recommend python packages on virtual environment(venv\).
- some python scripts are implemented at https://github.com/intel-innersource/libraries.ai.videoanalyticssuite.gpu-tools
  --> all python scripts will be moved to this run_daily repo.

# Trigger scripts
- trigger_download_ov_nightly.bat : get the latest ov nightly package
- trigger_build_apps.bat : build three apps for test. benchmark_app, llm for qwen/chatglm3
- trigget_run_daily.bat
