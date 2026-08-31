S=/data/lsh/mlperf_inference_storage
find "$S" -name '*8313*' 2>/dev/null
ls -la "$S/data/llama3.1-405b/"
ls -la "$S/preprocessed_data/llama3.1-405b/"



S=/data/lsh/mlperf_inference_storage
SRC=$(find "$S" -name 'mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl' | head -1)
cp "$SRC" "$S/preprocessed_data/llama3.1-405b/"
ls -lh "$S/preprocessed_data/llama3.1-405b/"




S=/data/lsh/mlperf_inference_storage
mkdir -p "$S/preprocessed_data/llama3.1-405b"
cd "$S/preprocessed_data/llama3.1-405b"
R2=https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh
bash <(curl -s $R2) https://inference.mlcommons-storage.org/metadata/llama3-1-405b-dataset-8313.uri
find . -mindepth 2 -name '*.pkl' -exec mv -n {} . \;
ls -lh



srun --gres=gpu:1 --time=00:10:00 \
  --container-image /data/lsh/inference_results_v6.0/closed/NVIDIA/build/sqsh_images/mlperf-inference-dell-x86_64-release.sqsh \
  --container-mounts /data/lsh/inference_results_v6.0/closed/NVIDIA:/work,/data/lsh/mlperf_inference_storage:/home/mlperf_inference_storage \
  --container-workdir /work --container-remap-root \
  --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
  bash -lc 'make link_dirs >/dev/null; ls -la /work/build/preprocessed_data/llama3.1-405b/'
