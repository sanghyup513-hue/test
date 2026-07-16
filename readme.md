/home/ubuntu/venv/mlcflow/bin/python3 /home/ubuntu/MLC/repos/mlcommons@mlperf-automations/script/get-generic-python-lib/detect-version.py
[2026-07-16 04:43:15,890 customize.py   : 152 INFO ] -                     Detected version: 4.4.4
[2026-07-16 04:43:15,897 script_utils.py:  88 INFO ] -               * mlcr get,python3
[2026-07-16 04:43:15,898 module.py      :1046 INFO ] -                    ! load /home/ubuntu/MLC/repos/local/cache/get-python3_38dfdb34/mlc-cached-state.json
[2026-07-16 04:43:15,898 module.py      :1046 INFO ] -                    ! load /home/ubuntu/MLC/repos/local/cache/get-generic-python-lib_7c1c2ffb/mlc-cached-state.json
make generate_engines RUN_ARGS=' --benchmarks=llama2-70b --scenarios=offline  --test_mode=PerformanceOnly  --gpu_batch_size="llama2-70b:4096" --log_dir=/home/ubuntu/MLC/repos/local/cache/get-mlperf-inference-results-dir_cb3f751a/test_results/buildkitsandbox-nvidia_original-gpu-tensorrt-vdefault-default_config/llama2-70b-99/offline/performance/run_1 --use_deque_limit --use_fp8 --checkpoint_dir=/home/ubuntu/MLC/repos/local/cache/get-mlperf-inference-nvidia-scratch-space_c4fdfe3a/models/Llama2/fp8-quantized-ammo/llama-2-70b-chat-hf-tp8pp1-fp8 --tensor_path=/home/ubuntu/MLC/repos/local/cache/get-mlperf-inference-nvidia-scratch-space_c4fdfe3a/preprocessed_data/open_orca --trtllm_build_flags=tensor_parallelism:8,pipeline_parallelism:1 --no_audit_verify  '
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/ubuntu/MLC/repos/local/cache/get-git-repo_mlperf-inferenc_3dd99c6f/repo/closed/NVIDIA/code/main.py", line 59, in <module>
    import code.ops as Ops
  File "/home/ubuntu/MLC/repos/local/cache/get-git-repo_mlperf-inferenc_3dd99c6f/repo/closed/NVIDIA/code/ops/__init__.py", line 16, in <module>
    from .generate_engines import (EngineBuilderOp,
  File "/home/ubuntu/MLC/repos/local/cache/get-git-repo_mlperf-inferenc_3dd99c6f/repo/closed/NVIDIA/code/ops/generate_engines.py", line 26, in <module>
    from ..common.workload import EngineIndex
  File "/home/ubuntu/MLC/repos/local/cache/get-git-repo_mlperf-inferenc_3dd99c6f/repo/closed/NVIDIA/code/common/workload.py", line 22, in <module>
    from nvmitten.nvidia.builder import TRTBuilder
  File "/home/ubuntu/venv/mlcflow/lib/python3.12/site-packages/nvmitten/nvidia/builder.py", line 27, in <module>
    import onnx_graphsurgeon as gs
ModuleNotFoundError: No module named 'onnx_graphsurgeon'
make: *** [Makefile:38: generate_engines] Error 1
[2026-07-16 04:43:26,329 main.py        : 137 ERROR] - Error during 'script' action: Script run execution failed in /home/ubuntu/MLC/repos/mlcommons@mlperf-automations/automation/script/module.py.
Error : Native run script failed inside MLC script (name = app-mlperf-inference-nvidia, return code = 256)


[2026-07-16 04:43:26,330 main.py        : 151 ERROR] -   at /home/ubuntu/venv/mlcflow/lib/python3.12/site-packages/mlc/script_action.py:345 in call_script_module_function
[2026-07-16 04:43:26,330 main.py        : 182 ERROR] - Failed script: app-mlperf-inference-nvidia
[2026-07-16 04:43:26,330 main.py        : 183 ERROR] - To rerun just the failed part: mlcr app-mlperf-inference-nvidia --print_env=False --print_deps=False --dump_version_info=True
[2026-07-16 04:43:26,330 main.py        : 205 ERROR] - Please file an issue at https://github.com/mlcommons/mlperf-automations/issues with the full console log.
[2026-07-16 04:43:26,330 main.py        : 209 ERROR] - mlcflow 1.2.5
[2026-07-16 04:43:26,336 main.py        : 214 ERROR] -   mlcommons@mlperf-automations: dev 66ee3bb5f
groups: cannot find name for group ID 993
ubuntu@5d003d01945e:~$
