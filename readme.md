### 사전 nvcr.io 로그인 필요
https://org.ngc.nvidia.com/account 에서 API Keys 생성 필요  



# ============ 기준 디렉터리 ============
export BASE=/data/lsh-test
export MLPERF_SCRATCH_PATH=$BASE/scratch # 모델/데이터/전처리 (≥10TB 권장)
export REPO=$BASE/inference_results_v5.0 # 코드 클론 위치
# =====================================

# 1) Dell 제출 코드 클론 (sparse: closed/Dell만)
mkdir -p "$BASE"
git clone --no-checkout --depth 1 --filter=blob:none \
  https://github.com/mlcommons/inference_results_v5.0.git "$REPO"
cd "$REPO"
git sparse-checkout init --cone
git sparse-checkout set --skip-checks closed/Dell
git checkout
cd closed/Dell

# 2) 스크래치 디렉터리 생성
mkdir -p "$MLPERF_SCRATCH_PATH"/{data,models,preprocessed_data}
mkdir -p "$MLPERF_SCRATCH_PATH"/data/llama2-70b

# 3) MLPerf 컨테이너 빌드 + 실행 (실행하면 컨테이너 "안"으로 진입)

vi Makefile.docker
PARTNER_DROP를 1 -> 0으로 변경
make prebuild

cd /work
make clean && make link_dirs
ls -al build/ # data/models/preprocessed_data 가 스크래치로의 심볼릭 링크여야 함
# --- 모델 + 데이터셋 받기 (MLCommons 멤버) ---
pip install mlc-scripts # 모델 -> build/models/Llama2/Llama-2-70b-chat-hf/ 
mlcr get,ml-model,llama2-70b,_pytorch,_r2-downloader,_70b,_mlc \ --outdirname=build/models/Llama2 -j

 # 전처리된 OpenOrca (validation + calibration pickle) 
 cd build/data/llama2-70b 
 bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \  https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri 
 
 gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz 2>/dev/null || true
gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz 2>/dev/null || true 
cd /work 

# --- 전처리 --- 
BENCHMARKS=llama2-70b make preprocess_data 

# --- 하네스 + TRT-LLM 빌드 (첫 빌드는 느림) --- 
make build 

# --- 엔진 생성 + 실행 (Offline, 99% 정확도 타겟) --- 
mkdir -p /work/build/models/Llama2/fp8-quantized-modelopt

make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=PP2"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --test_mode=PerformanceOnly --config_ver=PP2" 
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --test_mode=AccuracyOnly"



https://llama2.mlcommons-storage.org/cdn-cgi/access/cli?aud=2c232a285ef6d844dcaaa90dc7569f432fa890bfc956c2dc4cd2f3f0016768b7&edge_token_transfer=true&redirect_url=https%3A%2F%2Fllama2.mlcommons-storage.org%2Fmetadata%2Fllama-2-70b-chat-hf.uri%3Faud%3D2c232a285ef6d844dcaaa90dc7569f432fa890bfc956c2dc4cd2f3f0016768b7%26token%3DIXfBeZ4Q8dc_9VlXby43inPIhHFZbZuwh86nFccz6HI%253D&send_org_token=true&token=IXfBeZ4Q8dc_9VlXby43inPIhHFZbZuwh86nFccz6HI%


[2026-07-16 21:43:32,562 run_harness.py:169 INFO] Result: result_tokens_per_second: 31313.2, Result is VALID, 10-min runtime requirement met: True


-----------------------------

MLPerf 수행 명령어 공유 드립니다.

export BASE=/data/lsh-test
export MLPERF_SCRATCH_PATH=$BASE/scratch 
export REPO=$BASE/inference_results_v5.0


cd /data/lsh-test/inference_results_v5.0/closed/Dell
make prebuild
export PATH=$HOME/.local/bin:$PATH
SKIP_TRTLLM_BUILD=1 make build

## 밀리초 단위, 최소 2시간, max 3시간 실행
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --test_mode=PerformanceOnly --config_ver=PP2" 
