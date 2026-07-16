ubuntu@5d003d01945e:~$ mlcr app-mlperf-inference-nvidia --print-env=False --dump_version_info=True
[2026-07-16 04:53:34,211 script_utils.py:  88 INFO ] - * mlcr app-mlperf-inference-nvidia
[2026-07-16 04:53:34,211 main.py        : 137 ERROR] - Error during 'script' action: Script run execution failed in /home/ubuntu/MLC/repos/mlcommons@mlperf-automations/automation/script/module.py.
Error : no scripts were found with tags: ['app-mlperf-inference-nvidia'] (when variations ignored)
[2026-07-16 04:53:34,212 main.py        : 151 ERROR] -   at /home/ubuntu/venv/mlcflow/lib/python3.12/site-packages/mlc/script_action.py:345 in call_script_module_function
[2026-07-16 04:53:34,212 main.py        : 182 ERROR] - Failed script: app-mlperf-inference-nvidia
[2026-07-16 04:53:34,212 main.py        : 183 ERROR] - To rerun just the failed part: mlcr app-mlperf-inference-nvidia --print_env=False --dump_version_info=True
[2026-07-16 04:53:34,212 main.py        : 205 ERROR] - Please file an issue at https://github.com/mlcommons/mlperf-automations/issues with the full console log.
[2026-07-16 04:53:34,212 main.py        : 209 ERROR] - mlcflow 1.2.5
[2026-07-16 04:53:34,217 main.py        : 214 ERROR] -   mlcommons@mlperf-automations: dev 66ee3bb5f
ubuntu@5d003d01945e:~$
