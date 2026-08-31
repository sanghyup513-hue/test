S=/data/lsh/mlperf_inference_storage
echo "=== 전체 검색 ==="
find "$S" -name '*8313*' 2>/dev/null
echo "=== data 디렉토리 ==="
ls -la "$S/data/llama3.1-405b/"
echo "=== preprocessed 디렉토리 ==="
ls -la "$S/preprocessed_data/llama3.1-405b/"


S=/data/lsh/mlperf_inference_storage
SRC=$(find "$S" -name 'mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl' 2>/dev/null | head -1)
echo "찾은 경로: $SRC"

mkdir -p "$S/preprocessed_data/llama3.1-405b"
ln -sf "$SRC" "$S/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
ls -la "$S/preprocessed_data/llama3.1-405b/"


cp "$SRC" "$S/preprocessed_data/llama3.1-405b/"
ls -lh "$S/preprocessed_data/llama3.1-405b/"
