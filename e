dell@dell:/data/mlperf/llama3.1-405b/model$ cd /data/lsh/inference_results_v6.0/closed/NVIDIA
D=build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46
ls -la "$D"/B300-SXM-270GBx4_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json

srun --gres=gpu:1 --time=01:00:00 \
  --container-image /data/lsh/inference_results_v6.0/closed/NVIDIA/build/sqsh_images/mlperf-inference-dell-x86_64-release.sqsh \
  --container-mounts /data/lsh/inference_results_v6.0/closed/NVIDIA:/work,/data/lsh/mlperf_inference_storage:/home/mlperf_inference_storage \
  --container-workdir /work --container-remap-root \
  --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
  bash -lc "make link_dirs && make display_results LOG_DIR=/work/$D"
-rw-rw-r-- 1 dell dell 42917061 Aug 31 20:35 build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46/B300-SXM-270GBx4_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json


/bin/bash: line 1: [: -ge: unary operator expected
/bin/bash: line 1: [: -ge: unary operator expected
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/work/scripts/print_harness_result.py", line 195, in <module>
    all_acc_pass = print_session_results(log_dir)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scripts/print_harness_result.py", line 105, in print_session_results
    results = enumerate_results(base_dir)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scripts/print_harness_result.py", line 70, in enumerate_results
    acc_results = check_accuracy(wl)
                  ^^^^^^^^^^^^^^^^^^
  File "/work/code/common/mlcommons/accuracy_checker.py", line 951, in check_accuracy
    with (wl.log_dir / "mlperf_log_accuracy.json").open(mode='r') as lf:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46/B300-SXM-270GBx1_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json'
make: *** [Makefile:104: display_results] Error 1
srun: error: dell: task 0: Exited with exit code 2
dell@dell:/data/lsh/inference_results_v6.0/closed/NVIDIA$ MD=/data/lsh/mlperf_inference_storage/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct
ls -la "$MD" 2>/dev/null
total 8952
drwxrwxr-x 3 dell dell    4096 Aug 31 21:36 .
drwxrwxr-x 4 dell dell    4096 Aug 30 13:12 ..
drwxrwxr-x 3 dell dell    4096 Aug 31 21:36 .cache
-rw-rw-r-- 1 dell dell     183 Aug 31 21:36 generation_config.json
-rw-rw-r-- 1 dell dell     296 Aug 31 21:36 special_tokens_map.json
-rw-rw-r-- 1 dell dell   55351 Aug 31 21:36 tokenizer_config.json
-rw-rw-r-- 1 dell dell 9085657 Aug 31 21:36 tokenizer.json
dell@dell:/data/lsh/inference_results_v6.0/closed/NVIDIA$
[mlperf] 0:bash*                                             
