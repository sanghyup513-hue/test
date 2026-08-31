dell@dell:/data/lsh$ S=/data/lsh/mlperf_inference_storage
find "$S" -name '*8313*' 2>/dev/null
ls -la "$S/data/llama3.1-405b/"
ls -la "$S/preprocessed_data/llama3.1-405b/"
/data/lsh/mlperf_inference_storage/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
/data/lsh/mlperf_inference_storage/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
/data/lsh/mlperf_inference_storage/data/llama3.1-405b/llama3-1-405b-dataset-8313.md5
total 926368
drwxrwxr-x 2 dell dell      4096 Aug 30 13:28 .
drwxrwxr-x 4 dell dell      4096 Aug 30 13:12 ..
-rw-rw-r-- 1 dell dell       103 Aug 30 13:28 llama3-1-405b-calibration-dataset-512.md5
-rw-rw-r-- 1 dell dell        92 Aug 30 13:24 llama3-1-405b-dataset-8313.md5
-rw-rw-r-- 1 dell dell  56687800 Jan  2  2025 mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl
-rw-rw-r-- 1 dell dell 891889920 Jan  2  2025 mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
total 649512
drwxrwxr-x 3 dell dell      4096 Aug 31 21:47 .
drwxrwxr-x 4 dell dell      4096 Aug 30 13:12 ..
-rw-rw-r-- 1 dell dell 665040128 Aug 30 14:03 input_ids_padded.npy
-rw-rw-r-- 1 dell dell     33380 Aug 30 14:03 input_lens.npy
drwxrwxr-x 2 dell dell      4096 Aug 30 14:03 mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval
lrwxrwxrwx 1 dell dell       124 Aug 31 21:47 mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl -> /data/lsh/mlperf_inference_storage/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
