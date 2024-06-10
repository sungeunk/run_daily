@echo off && setlocal

call init_env.bat

call trigger_download_models.bat
call trigger_generate_token.bat
call trigger_download_ov_nightly.bat
call trigger_build_apps.bat
call trigger_convert_models.bat
call trigger_run_llm_daily.bat