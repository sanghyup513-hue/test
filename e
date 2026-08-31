make: *** [Makefile:104: display_results] Error 1
srun: error: dell: task 0: Exited with exit code 2
dell@dell:/data/lsh/inference_results_v6.0/closed/NVIDIA$ cd /data/lsh/inference_results_v6.0/closed/NVIDIA
D=build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46
srun --gres=gpu:1 --time=01:00:00 \
  --container-image /data/lsh/inference_results_v6.0/closed/NVIDIA/build/sqsh_images/mlperf-inference-dell-x86_64-release.sqsh \
  --container-mounts /data/lsh/inference_results_v6.0/closed/NVIDIA:/work,/data/lsh/mlperf_inference_storage:/home/mlperf_inference_storage \
  --container-workdir /work --container-remap-root \
  --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
  bash -lc "make link_dirs && make display_results LOG_DIR=/work/$D SYSTEM_NAME=B300-SXM-270GBx4"
/bin/bash: line 1: [: -ge: unary operator expected
/bin/bash: line 1: [: -ge: unary operator expected
[2026-08-31 21:52:12,595 utils.py:55 INFO RANK=0] Running command: /usr/bin/python3 /work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py --checkpoint-path /work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct --mlperf-accuracy-file /work/build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46/B300-SXM-270GBx4_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json --dataset-file /work/build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl --dtype int32
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt_tab.zip.

100%|██████████| 8313/8313 [06:08<00:00, 22.55it/s]

Results

{'exact_match': 90.07510258107214, 'rougeL': 21.839943978781175, 'gen_len': 22312083, 'gen_num': 8313, 'gen_tok_len': 5289026, 'tokens_per_sample': 636.2}

======================== Result summaries: ========================

Offline Scenario:
+----------------------+---------------+-----------+------------------+-------------------+------------------+-------------+
| System Name          | Benchmark     | Setting   | All Acc. Pass?   | Metric Name       |   Measured Value | Threshold   |
+======================+===============+===========+==================+===================+==================+=============+
| B300-SXM-270GBx4_TRT | llama3.1-405b | cp990     | Yes              | ROUGEL            |            21.84 | >=21.449934 |
|                      |               |           |                  | exact_match       |            90.08 | >=89.232165 |
|                      |               |           |                  | TOKENS_PER_SAMPLE |           636.20 | >=616.212   |
+----------------------+---------------+-----------+------------------+-------------------+------------------+-------------+
dell@dell:/data/lsh/inference_results_v6.0/closed/NVIDIA$
[mlperf] 0:bash*                                                                                                                                      "dell" 22:00 31-Aug-26
