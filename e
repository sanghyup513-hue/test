ls -la /data/lsh/mlperf_inference_storage/data/llama3.1-405b/
ls -la /data/lsh/mlperf_inference_storage/preprocessed_data/llama3.1-405b/


S=/data/lsh/mlperf_inference_storage
F=mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl

find "$S" -name "$F" 2>/dev/null

# 위에서 찾은 경로를 확인한 뒤
ln -sf "$S/data/llama3.1-405b/$F" "$S/preprocessed_data/llama3.1-405b/$F"
ls -la "$S/preprocessed_data/llama3.1-405b/$F"


cd /data/lsh/inference_results_v6.0/closed/NVIDIA
D=build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46
srun --gres=gpu:1 --time=01:00:00 \
  --container-image /data/lsh/inference_results_v6.0/closed/NVIDIA/build/sqsh_images/mlperf-inference-dell-x86_64-release.sqsh \
  --container-mounts /data/lsh/inference_results_v6.0/closed/NVIDIA:/work,/data/lsh/mlperf_inference_storage:/home/mlperf_inference_storage \
  --container-workdir /work --container-remap-root \
  --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
  bash -lc "make link_dirs && make display_results LOG_DIR=/work/$D SYSTEM_NAME=B300-SXM-270GBx4"
