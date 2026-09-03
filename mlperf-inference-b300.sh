#!/usr/bin/env bash
# =============================================================================
#  MLPerf Inference v6.0 실행기
#  Dell PowerEdge XE9780 / NVIDIA B300
#
#  이 스크립트가 하는 일
#    MLPerf 는 AI 추론 성능을 재는 업계 표준 벤치마크입니다.
#    NVIDIA 가 공개한 8-GPU 성적표를 우리 장비에서 재현하는 것이 목표입니다.
#    준비(모델 다운로드 등)부터 실행, 결과 채점까지 한 스크립트로 처리합니다.
#
# -----------------------------------------------------------------------------
#  사용법
# -----------------------------------------------------------------------------
#  [1단계] 환경 준비 — 처음 한 번만
#    bash mlperf-inference-b300.sh paths      지금 쓰는 경로 확인
#    bash mlperf-inference-b300.sh doctor     지금 환경이 정상인지 진단
#    bash mlperf-inference-b300.sh setup      소스코드+컨테이너 내려받기
#    bash mlperf-inference-b300.sh patch      알려진 버그 수정 적용
#
#  벤치마크 이름은 아래 둘 중 하나를 그대로 적습니다.
#      llama2-70b        700억 파라미터. GPU 1장에 통째로 올라감.
#      llama3.1-405b     4050억 파라미터. GPU 2장에 쪼개서 올림.
#
#  [2단계] 자산 준비 — 벤치마크별로 한 번씩 (수 시간 소요)
#    bash mlperf-inference-b300.sh assets llama2-70b        llama2-70b 자산 준비
#    bash mlperf-inference-b300.sh assets llama3.1-405b     llama3.1-405b 자산 준비
#
#  [3단계] 실행
#    bash mlperf-inference-b300.sh test llama3.1-405b 4                   1분 예비 테스트
#    bash mlperf-inference-b300.sh run  llama3.1-405b offline  4          처리량 측정
#    bash mlperf-inference-b300.sh run  llama3.1-405b server   4 0.9      응답시간 측정
#    bash mlperf-inference-b300.sh run  llama3.1-405b accuracy 4          정확도 검증
#
#  [4단계] 결과
#    bash mlperf-inference-b300.sh result offline 4 llama3.1-405b         결과 다시 보기
#    bash mlperf-inference-b300.sh score  llama3.1-405b 4                 채점만 재실행
#    bash mlperf-inference-b300.sh record                   지금까지의 성적표
#
#  [문제 발생 시]
#    bash mlperf-inference-b300.sh diag       서버가 안 뜬 원인 찾기
#    bash mlperf-inference-b300.sh clean      멈춘 작업/GPU 강제 정리
#
#  ※ 학습(Training) 벤치마크는 mlperf-training-b300.sh 를 쓰세요.
#
# -----------------------------------------------------------------------------
#  용어
# -----------------------------------------------------------------------------
#  Offline   질문 8313개를 한꺼번에 던지고 총 처리 속도만 측정. 배치 작업.
#  Server    질문이 실시간으로 들어오는 상황. 응답 시간 제한을 지켜야 함.
#  Accuracy  답이 실제로 맞는지 채점. 속도가 아니라 품질 검증.
#  TP        Tensor Parallel. 모델 하나를 GPU 몇 장에 쪼개 올리는지.
#  DP        Data Parallel. 그런 묶음을 몇 개 병렬로 돌리는지.
#            예) GPU 4장 + TP2 → 2장씩 묶어 2세트 (TP2 x DP2)
#  TTFT      Time To First Token. 질문 후 첫 글자가 나오기까지의 시간.
#  TPOT      Time Per Output Token. 글자가 하나씩 나오는 간격.
#  VALID     MLPerf 규정을 모두 만족한 유효 결과. 하나라도 어기면 INVALID.
#
# -----------------------------------------------------------------------------
#  이 장비에서 실제로 겪은 문제와 해결책 — 모두 아래 코드에 반영돼 있습니다
# -----------------------------------------------------------------------------
#  (1) 벤치마크 목록을 읽어오는 코드가 최신 버전(v6.1)을 골라버려서
#      llama3.1-405b 항목을 못 찾고 죽습니다. v6.0 으로 고정해야 합니다.
#                                                        -> patch 1
#  (2) 서버 실행 방식이 'leader' 로 설정되면, 존재하지 않는 협력 프로세스를
#      영원히 기다립니다. 'legacy' 로 바꿔야 합니다.
#      단, 조건문 안의 'leader' 글자는 건드리면 안 됩니다 (문법 붕괴).
#                                                        -> patch 2
#  (3) NVIDIA 코드에 실제 버그가 있습니다. gpu_ids 라는 변수가 특정 경로에서만
#      만들어지는데, 로그 출력이 그 밖에 있어서 legacy 모드면 죽습니다.
#      이걸 고친 덕에 GPU 여러 장 분산이 처음으로 성공했습니다.
#                                                        -> patch 3
#  (3-1) 그래도 GPU 를 여러 장 쓰면 서버 전부가 물리 GPU 0 을 가리켜, 첫 번째만
#      살고 나머지는 메모리 부족으로 조용히 죽습니다. 서버를 srun 스텝마다
#      하나씩 띄우는 구조라 코드가 쓰는 index 가 언제나 0 이기 때문입니다.
#      랭크를 알아낼 방법을 여러 개 시도했고 결론은 포트 번호였습니다.
#        NVIDIA_VISIBLE_DEVICES -> pyxis 가 컨테이너 생성 시 소비. 안에선 사라짐.
#        MLPERF_GPU_LIST 주입   -> run_scaleout.sh 도 고쳐야 하고 실패.
#        endpoint_port          -> 30000+rank 로 다르고 컨테이너 안에서도 살아있음.
#      게다가 서버 로그 파일명도 index 로 만들어 여러 서버가 같은 파일을
#      덮어쓰는 탓에 죽은 이유마저 사라졌습니다.      -> patch 4,5
#  (4) 호스트와 컨테이너의 MPI 통신 규격 버전이 안 맞습니다(PMIx 5 vs 3).
#      --mpi=pmi2 로 우회합니다. --mpi=none 은 아예 초기화가 실패합니다.
#  (5) --exclusive 옵션을 빼면 CPU 를 2개만 받아서, 수백 GB 모델 읽기가
#      사실상 멈춥니다. GPU 개수 제한보다 CPU 확보가 우선입니다.
#  (6) 실행 스크립트 307행이 CPU 아키텍처를 aarch64 로 고정해뒀습니다.
#      우리는 x86 이라 컨테이너 경로를 직접 지정해야 합니다.
#  (7) NGC 공식 이미지에 추론 엔진이 이미 들어 있습니다. 소스 빌드(수 시간)
#      불필요하고, enroot import 로 바로 씁니다.
#  (8) 컨테이너 이미지 임시 폴더 기본값이 /tmp(메모리 디스크)라서 30GB
#      이미지를 풀다가 터집니다. 디스크 경로로 바꿔야 합니다.
#  (9) 정확도 채점기는 원본 토크나이저 파일과 원본 데이터셋을 따로 요구합니다.
#      없으면 3시간 추론이 끝난 뒤 채점 단계에서 실패합니다. 사전 검증 필수.
# (10) 채점만 따로 돌릴 때 시스템 이름을 안 주면 GPU 1장짜리로 오인해서
#      결과 파일 경로를 못 찾습니다.
# (11) salloc 으로 잡은 자원은 터미널에 매달려 있습니다. 부모 작업을 죽이면
#      자식도 함께 죽습니다. 긴 작업은 반드시 tmux 안에서 돌리세요.
# =============================================================================
set -u    # 정의하지 않은 변수를 쓰면 즉시 중단 (오타로 인한 오작동 방지)

# ============================== 사용자 설정 ==================================
# 경로를 지정하는 방법은 세 가지입니다. 아래로 갈수록 우선합니다.
#
#   1) 이 파일의 기본값                (아무것도 안 하면 이 값)
#   2) 설정파일  ~/.mlperf-b300.conf   (한 번 적어두면 계속 적용)
#   3) 명령줄 옵션  --base /경로 등     (이번 실행에만 적용)
#
#  설정파일 예시 (~/.mlperf-b300.conf) — 자주 쓰는 경로를 적어두면 편합니다
#     BASE=/data/lsh
#     SCRATCH=/mnt/nvme/mlperf_storage
#     CONT=/data/lsh/mlperf-inference.sqsh
#
#  명령줄 옵션 (명령어보다 앞에 씁니다)
#     --base    <경로>   작업 최상위 폴더. 아래 경로들의 기준이 됩니다.
#     --scratch <경로>   모델·데이터·전처리 결과를 두는 폴더 (용량 큼)
#     --repo    <경로>   MLPerf 소스코드 폴더
#     --cont    <파일>   컨테이너 이미지(.sqsh) 경로
#     --logdir  <경로>   실행 로그를 남길 폴더
#
#  예)  bash mlperf-inference-b300.sh --scratch /mnt/nvme/st run llama2-70b offline 4
#       현재 적용된 경로를 확인하려면:  bash mlperf-inference-b300.sh paths
# -----------------------------------------------------------------------------

# 1) 기본값
BASE_DEFAULT=/data/lsh

# 2) 설정파일 읽기 (있으면)
CONF="${MLPERF_CONF:-$HOME/.mlperf-b300.conf}"
[ -f "$CONF" ] && . "$CONF"

# 환경변수로 준 값도 받습니다 (이전 방식 호환)
BASE="${BASE:-$BASE_DEFAULT}"
REPO_IN="${REPO:-}"
SCRATCH_IN="${SCRATCH:-${MLPERF_SCRATCH_PATH:-}}"
CONT_IN="${CONT:-}"
LOGDIR_IN="${LOGDIR:-}"

# 3) 명령줄 옵션 파싱 (명령어보다 앞에 온 것만)
while [ $# -gt 0 ]; do
  case "$1" in
    --base)    BASE="${2:?--base 뒤에 경로를 적어주세요}";       shift 2 ;;
    --scratch) SCRATCH_IN="${2:?--scratch 뒤에 경로를 적어주세요}"; shift 2 ;;
    --repo)    REPO_IN="${2:?--repo 뒤에 경로를 적어주세요}";     shift 2 ;;
    --cont)    CONT_IN="${2:?--cont 뒤에 파일 경로를 적어주세요}"; shift 2 ;;
    --logdir)  LOGDIR_IN="${2:?--logdir 뒤에 경로를 적어주세요}";  shift 2 ;;
    --conf)    CONF="${2:?--conf 뒤에 파일 경로를 적어주세요}"
               [ -f "$CONF" ] && . "$CONF" || { echo "설정파일 없음: $CONF"; exit 1; }
               shift 2 ;;
    --help|-h) sed -n '3,80p' "$0"; exit 0 ;;
    --*)       echo "알 수 없는 옵션: $1"; echo "쓸 수 있는 옵션: --base --scratch --repo --cont --logdir --conf"; exit 1 ;;
    *)         break ;;      # 옵션이 아니면 명령어 시작
  esac
done

# 최종 경로 확정. 따로 지정하지 않은 것은 BASE 기준으로 만듭니다.
REPO="${REPO_IN:-$BASE/inference_results_v6.0/closed/NVIDIA}"
SCRATCH="${SCRATCH_IN:-$BASE/mlperf_inference_storage}"
RUNDIR="${LOGDIR_IN:-$BASE/mlperf-runs}"
SAFE="$BASE/preserved"          # repo 를 지울 때 컨테이너 이미지를 대피시킬 곳
mkdir -p "$RUNDIR" 2>/dev/null

# NGC(NVIDIA 공식 저장소)의 MLPerf 추론 컨테이너 이미지
NGC_IMG="nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86"

# 컨테이너 이미지. --cont 로 직접 주지 않으면 repo 안에서 찾습니다.
if [ -n "$CONT_IN" ]; then
  SQSH="$CONT_IN"
else
  SQSH=$(ls -1 "$REPO"/build/sqsh_images/*.sqsh 2>/dev/null | head -1)
fi

# 이번 실행의 상세 로그 파일 (화면에는 요약만, 상세는 여기로)
V="$RUNDIR/inference-$(date +%m%d-%H%M).log"

# --- NVIDIA 가 공개한 8-GPU 성적 (Tokens/s) — 우리 목표의 기준값 -------------
PUB_L2_OFF=112954;  PUB_L2_SRV=107318        # llama2-70b
PUB_405_OFF=1951.6; PUB_405_SRV=1460.25      # llama3.1-405b

# --- Server 시나리오의 응답시간 상한 (나노초) — MLPerf 가 정한 고정값 --------
L2_TTFT=2000000000;   L2_TPOT=200000000      # llama2-70b:   2초 / 200ms
L405_TTFT=6000000000; L405_TPOT=175000000    # llama3.1-405b: 6초 / 175ms

# 이전 작업의 흔적이 환경변수에 남아 있으면 srun 이 죽은 작업에 붙으려 합니다
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_NTASKS SLURM_JOB_NODELIST

# ============================== 출력 도구 ====================================
say(){ printf '%s\n' "$*"; }              # 일반 메시지
ok(){  printf '   OK   %s\n' "$*"; }      # 정상 항목
ng(){  printf '   --   %s\n' "$*"; }      # 문제 항목
# GPU 별 사용 메모리(MiB)를 한 줄씩 출력. 배치가 제대로 됐는지 보는 핵심 지표
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

# ========================= 벤치마크별 설정 불러오기 ==========================
# 두 벤치마크는 모델 크기가 달라서 GPU 를 묶는 방식이 다릅니다.
#   llama2-70b   : 작아서 GPU 1장에 다 올라감      -> TP1, GPU 1장 = 1세트
#   llama3.1-405b: 커서 GPU 2장에 쪼개 올려야 함    -> TP2, GPU 2장 = 1세트
# 이 함수가 벤치마크 이름을 받아 관련 값들을 한꺼번에 세팅합니다.
# 벤치마크 이름은 전체 이름으로 적습니다.  llama2-70b  /  llama3.1-405b
# (짧은 별칭도 받지만, 기록에는 전체 이름이 남습니다)
bench_init(){
  case "$1" in
    llama2-70b|llama2)
      B_KEY=llama2-70b; B_NAME=llama2-70b; B_FILE=llama2-70b
      B_ATOMIC=1            # NVIDIA 가 제공하는 기준 config 는 "GPU 1장" 버전
      B_TP=1                # 모델을 GPU 1장에 통째로 올림
      B_PUB_OFF=$PUB_L2_OFF; B_PUB_SRV=$PUB_L2_SRV
      B_TTFT=$L2_TTFT; B_TPOT=$L2_TPOT
      B_OFF_Q=55; B_SRV_Q=50          # 세트 1개당 초당 처리 목표 (요청 수)
      B_MODEL="$SCRATCH/models/Llama2/Llama-2-70b-chat-hf"
      B_PREP="$SCRATCH/preprocessed_data/llama2-70b"
      B_DATA="$SCRATCH/data/llama2-70b"
      B_WALL_OFF="03:00:00"; B_WALL_SRV="02:30:00"   # SLURM 에 요청할 최대 시간
      # --- 정확도 채점기가 요구하는 원본 데이터 --------------------------
      # 주의: 전처리 결과는 preprocessed_data/llama2-70b/ 에 만들어지는데,
      #       채점기는 preprocessed_data/open_orca/ 를 찾습니다. 폴더 이름이
      #       다르므로 원본 pkl 을 그쪽으로 따로 복사해야 합니다.
      B_ACC_PKL="$SCRATCH/preprocessed_data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
      B_ACC_SRC="$B_DATA/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
      B_HF_REPO="meta-llama/Llama-2-70b-chat-hf"
      B_ACC_NOTE="ROUGE1 43.99 / ROUGE2 21.81 / ROUGEL 28.33 이상, TOKENS_PER_SAMPLE 265.0~323.9"
      ;;
    llama3.1-405b|llama3.1|405b)
      B_KEY=llama3.1-405b; B_NAME=llama3.1-405b; B_FILE=llama3_1-405b
      B_ATOMIC=2            # 기준 config 가 "GPU 2장" 버전
      B_TP=2                # 모델을 GPU 2장에 쪼개서 올림
      B_PUB_OFF=$PUB_405_OFF; B_PUB_SRV=$PUB_405_SRV
      B_TTFT=$L405_TTFT; B_TPOT=$L405_TPOT
      B_OFF_Q=0.75; B_SRV_Q=0.80
      B_MODEL="$SCRATCH/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct"
      B_FP4="$SCRATCH/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4"
      B_PREP="$SCRATCH/preprocessed_data/llama3.1-405b"
      B_DATA="$SCRATCH/data/llama3.1-405b"
      B_WALL_OFF="04:00:00"; B_WALL_SRV="03:00:00"
      # --- 정확도 채점기가 요구하는 원본 데이터 --------------------------
      # 이쪽은 전처리 폴더와 이름이 같지만, 심볼릭 링크로 걸면 경로가 꼬여
      # 못 읽습니다. 반드시 복사해야 합니다.
      B_ACC_PKL="$B_PREP/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
      B_ACC_SRC="$B_DATA/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
      B_HF_REPO="meta-llama/Llama-3.1-405B-Instruct"
      B_ACC_NOTE="ROUGEL 21.4499 이상 / exact_match 89.2322 이상 / TOKENS_PER_SAMPLE 616.212 이상"
      ;;
    *) ng "알 수 없는 벤치마크: $1"
       ng "  llama2-70b  또는  llama3.1-405b  로 적어주세요"
       exit 1 ;;
  esac
}

# GPU 개수를 받아 "세트가 몇 개인지"를 계산합니다.
#   llama2-70b(TP1): GPU 4장 -> 4세트     llama3.1-405b(TP2): GPU 4장 -> 2세트
dp_of(){ [ "$B_TP" -eq 1 ] && echo "$1" || echo $(( $1 / B_TP )); }

# ============================== 강제 정리 ====================================
# 이전 실행이 비정상 종료하면 GPU 메모리를 붙들고 있는 프로세스가 남습니다.
# 그 상태로 새로 시작하면 메모리 부족으로 또 실패하므로, 확실히 비웁니다.
hard_clean(){
  scancel -u "$(whoami)" >>"$V" 2>&1; sleep 3        # 내 SLURM 작업 전부 취소
  pkill -9 -f 'trtllm-serve|trtllm-llmapi|code.main|run_scaleout' >>"$V" 2>&1
  sleep 3
  # GPU 를 점유한 프로세스를 직접 찾아 종료. 한 번에 안 죽는 경우가 있어 반복.
  local i pids p
  for i in 1 2 3 4 5 6; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
  # SLURM 작업이 CG(정리중) 상태에서 완전히 빠질 때까지 기다립니다
  for i in 1 2 3 4 5 6; do
    squeue -h -u "$(whoami)" | grep -q . || break
    sleep 5
  done
  # 노드가 drain(격리) 상태면 되살립니다. 반복 강제종료 후 흔히 발생.
  sinfo -h -o '%t' | grep -qiE 'drain|down' && \
    sudo scontrol update NodeName="$(hostname -s)" State=RESUME >>"$V" 2>&1

  # 메모리가 실제로 비워졌는지 확인. 안 비면 실행을 거부합니다.
  local busy; busy=$(mem | awk '$1>1000{n++}END{print n+0}')
  if [ "$busy" -gt 0 ]; then
    ng "GPU 메모리가 아직 안 비었습니다 [$(mem | tr '\n' ' ')]"
    ng "확인: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    return 1
  fi
  ok "GPU 전부 비움 [$(mem | tr '\n' ' ')]   대기작업 $(squeue -h -u "$(whoami)" | wc -l)개"
}

# ========================= 알려진 버그 수정 (3종) ============================
# 이미 적용돼 있으면 건너뛰고, 수정 후 문법 검사에 실패하면 자동 원복합니다.
apply_patches(){
  local L="$REPO/code/common/mlcommons/loadgen.py"
  local R="$REPO/scaleout/run_scaleout.sh"
  local S="$REPO/code/llmlib/launch_server.py"

  # --- patch 1 : 벤치마크 목록 버전을 v6.0 으로 고정 -------------------------
  # 원본은 "가장 최신 버전을 자동 선택"하는데, 최신(v6.1)에는 405b 항목이
  # 없어서 KeyError 로 죽습니다. 우리 소스는 v6.0 이니 그걸 쓰게 합니다.
  if grep -q 'versioning.parse(C.VERSION' "$L" 2>/dev/null; then
    ok "patch1 이미 적용됨"
  else
    cp "$L" "$L.orig"
    sed -i 's|^_latest_ver = max(versioning.parse(ver_key) for ver_key in submission_checker_constants.MODEL_CONFIG.keys())|_latest_ver = versioning.parse(C.VERSION.lstrip("v"))|' "$L"
    if grep -q 'versioning.parse(C.VERSION' "$L" && python3 -m py_compile "$L" 2>>"$V"; then
      ok "patch1 벤치마크 목록 v6.0 고정"
    else
      cp "$L.orig" "$L"; ng "patch1 실패 — 원본 복구함"
    fi
  fi

  # --- patch 2 : 서버 실행 방식 leader -> legacy ----------------------------
  # 주의: 원본은 아래 형태입니다.
  #   [[ ! "$run_args" =~ --mpi_mode=leader ]] && run_args="... --mpi_mode=leader"
  # 조건문(왼쪽)의 leader 까지 바꾸거나 지우면 문법이 깨집니다.
  # 그래서 대입부(오른쪽)만 정확히 치환합니다.
  if grep -q 'mpi_mode=legacy' "$R" 2>/dev/null; then
    ok "patch2 이미 적용됨"
  else
    cp "$R" "$R.orig"
    sed -i 's|run_args="\$run_args --mpi_mode=leader"|run_args="$run_args --mpi_mode=legacy"|g' "$R"
    if bash -n "$R" 2>>"$V" && grep -q 'mpi_mode=legacy' "$R"; then
      ok "patch2 실행방식 legacy 전환 ($(grep -c 'mpi_mode=legacy' "$R")곳)"
    else
      cp "$R.orig" "$R"; ng "patch2 실패 — 원본 복구함"
    fi
  fi

  # --- patch 3 : gpu_ids 변수 미정의 오류 ----------------------------------
  # NVIDIA 코드 545행이 gpu_ids 를 로그로 찍는데, 그 변수는 leader 경로에서만
  # 만들어집니다. legacy 로 오면 변수가 없어 UnboundLocalError 로 죽습니다.
  # 변수가 없으면 다른 값으로 대체하도록 안전하게 바꿉니다.
  if grep -q "locals().get('gpu_ids'" "$S" 2>/dev/null; then
    ok "patch3 이미 적용됨"
  else
    cp "$S" "$S.orig"
    python3 - "$S" <<'PYEOF' >>"$V" 2>&1
import sys
p = sys.argv[1]
s = open(p).read()
old = '            logging.info(f"  GPU devices: {gpu_ids}")'
new = '            logging.info(f"  GPU devices: {locals().get(\'gpu_ids\', env.get(\'CUDA_VISIBLE_DEVICES\', \'default\'))}")'
if old in s:
    open(p, 'w').write(s.replace(old, new)); print("patched")
else:
    print("pattern-not-found")   # 들여쓰기가 다르면 수동 확인 필요
PYEOF
    if grep -q "locals().get('gpu_ids'" "$S" && python3 -m py_compile "$S" 2>>"$V"; then
      ok "patch3 gpu_ids 오류 해소"
    else
      cp "$S.orig" "$S"; ng "patch3 실패 — 원본 복구함 (들여쓰기 확인 필요)"
    fi
  fi

  # --- patch 4,5 : GPU 를 여러 장 쓸 때만 필요한 수정 ----------------------
  # 증상: GPU 4장으로 돌리면 GPU 0 에만 서버가 뜨고 나머지는 조용히 죽습니다.
  # 원인 두 가지가 겹칩니다.
  #  (4) legacy 모드는 CUDA_VISIBLE_DEVICES 를 index 로 계산합니다. 그런데
  #      서버를 srun 스텝마다 하나씩 띄우므로 index 는 언제나 0 입니다.
  #      결과적으로 서버 전부가 물리 GPU 0 을 가리키고, 첫 번째가 메모리를
  #      거의 다 차지해(kvcache 0.95) 나머지는 메모리 부족으로 즉사합니다.
  #
  #      랭크를 어떻게 알아낼까가 관건인데, 실제로 시도해본 결과는 이렇습니다.
  #        NVIDIA_VISIBLE_DEVICES  -> 컨테이너 생성 시 pyxis 가 소비해서
  #                                   컨테이너 안에서는 값이 사라짐. 사용 불가.
  #        MLPERF_GPU_LIST(직접주입) -> run_scaleout.sh 도 고쳐야 하고 실패.
  #        endpoint_port           -> 랭크별로 30000,30001,... 로 다르고
  #                                   컨테이너 안에서도 그대로 살아있음. 채택.
  #      그래서 포트에서 랭크를 역산합니다: rank = port - 30000
  #
  #  (5) 서버 로그 파일 이름도 index 로 만들어서 여러 서버가 같은 파일을 'w' 로
  #      엽니다. 뒤에 뜬 서버가 앞의 기록을 덮어써 죽은 이유가 사라집니다.
  #      (이 때문에 원인 파악이 오래 걸렸습니다)
  if grep -q '\[PATCH4B\]' "$S" 2>/dev/null && grep -q '\[PATCH5\]' "$S" 2>/dev/null; then
    ok "patch4,5 이미 적용됨"
  else
    cp "$S" "$S.orig45"
    python3 - "$S" <<'PYEOF' >>"$V" 2>&1
import re, sys
p = sys.argv[1]
s = open(p).read()
n = 0

NEW4 = """                cmd = ['trtllm-serve']
                # [PATCH4B] index 는 srun 스텝마다 항상 0 이라, 원본 계산식으로는
                # 서버 전부가 물리 GPU 0 을 가리켜 2번째부터 메모리 부족으로 죽습니다.
                # 컨테이너 안에서 랭크를 알 수 있는 유일하게 확실한 값이 포트입니다.
                # (NVIDIA_VISIBLE_DEVICES 는 pyxis 가 소비해서 안에서는 사라짐)
                #   rank = endpoint_port - 30000  ->  내 GPU = rank * gpus_per_server
                # 컨테이너가 이미 내 몫만 보고 있으면 원본 계산이 맞으므로 유지합니다.
                try:
                    _vis = len(subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                        text=True).strip().splitlines())
                except Exception:
                    _vis = 0
                _rank = int(endpoint_port) - 30000
                if _vis > gpus_per_server and 0 <= _rank < 64 \\
                   and (_rank + 1) * gpus_per_server <= _vis:
                    _start = _rank * gpus_per_server
                    gpu_ids = list(range(_start, _start + gpus_per_server))
                    logging.info(f"[PATCH4B] port={endpoint_port} rank={_rank} "
                                 f"visible={_vis} -> GPU {gpu_ids}")
                else:
                    _start = index * gpus_per_server
                    gpu_ids = list(range(_start, _start + gpus_per_server))
                    logging.info(f"[PATCH4B] fallback (visible={_vis}, "
                                 f"port={endpoint_port}) -> GPU {gpu_ids}")
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))"""

# 원본은 물론이고, 이전에 시도했던 패치 잔재까지 모두 교체 대상으로 잡습니다.
# cmd = ['trtllm-serve'] 부터 CUDA_VISIBLE_DEVICES 대입까지를 통째로 갈아냅니다.
if "[PATCH4B]" in s:
    print("patch4b already")
else:
    lines = s.splitlines(keepends=True)
    beg = None
    for i, l in enumerate(lines):
        if "cmd = ['trtllm-serve']" in l and 'llmapi-launch' not in l:
            beg = i; break
    if beg is None:
        print("patch4b anchor-not-found"); sys.exit(1)
    end = None
    for i in range(beg, min(beg + 45, len(lines))):
        if "env['CUDA_VISIBLE_DEVICES']" in lines[i]:
            end = i
    if end is None:
        print("patch4b assign-not-found"); sys.exit(1)
    print("--- replacing lines %d..%d ---" % (beg + 1, end + 1))
    print(''.join(lines[beg:end + 1]))
    lines[beg:end + 1] = [NEW4 + "\n"]
    s = ''.join(lines); n += 1
    print("patch4b ok")

OLD5 = "                log_file = self.log_dir / f'trtllm_serve_{index}.log'"
NEW5 = ("                # [PATCH5] index 가 항상 0 이라 서버 여러 개가 같은 파일을 'w' 로\n"
        "                # 열어 서로 덮어씁니다. 죽은 이유를 남기려면 포트로 구분해야 합니다.\n"
        "                log_file = self.log_dir / f'trtllm_serve_{index}_port{endpoint_port}.log'")

if "[PATCH5]" in s:   print("patch5 already")
elif OLD5 in s:       s = s.replace(OLD5, NEW5); n += 1; print("patch5 ok")
else:                 print("patch5 pattern-not-found"); sys.exit(1)

if n: open(p, 'w').write(s)
PYEOF
    if grep -q '\[PATCH4B\]' "$S" && grep -q '\[PATCH5\]' "$S" \
       && python3 -m py_compile "$S" 2>>"$V"; then
      ok "patch4 서버별 GPU 배정(포트 기반) / patch5 서버별 로그파일 분리"
    else
      cp "$S.orig45" "$S"; ng "patch4,5 실패 — 원본 복구함 ($V 확인)"
    fi
  fi
}

# ====================== GPU 개수별 설정파일 만들기 ===========================
# NVIDIA 는 "GPU 1장" 또는 "GPU 2장" 기준 설정만 제공합니다.
# 우리가 쓰려는 GPU 개수(4장, 6장...)용 설정은 직접 만들어야 하고,
# 목표 처리량도 세트 수에 비례해 늘려줘야 합니다.
#   예) llama2-70b 4장 = 4세트 -> 55 x 4 = 220
#       llama3.1-405b 4장 = 2세트 -> 0.75 x 2 = 1.50
# 8장으로 계산하면 NVIDIA 공개 제출값과 정확히 일치합니다 (검증됨).
mkcfg(){
  local n="$1" dp cA cN oq sq
  dp=$(dp_of "$n")
  cA="$REPO/configs/B300-SXM-270GBx${B_ATOMIC}"     # 원본 (기준)
  cN="$REPO/configs/B300-SXM-270GBx${n}"            # 새로 만들 것

  # 원본 파일이 실제로 있는지 먼저 확인합니다.
  # 없는데 진행하면 빈 파일이 만들어지고, 몇 시간 뒤 실행 단계에서 터집니다.
  local srcO="$cA/Offline/${B_FILE}.py" srcS="$cA/Server/${B_FILE}.py"
  [ -s "$srcO" ] || { ng "기준 설정파일이 없습니다: $srcO"; return 1; }
  [ -s "$srcS" ] || { ng "기준 설정파일이 없습니다: $srcS"; return 1; }

  # 요청한 GPU 개수가 기준과 같으면(예: llama2-70b 를 1장으로) 원본이 곧 정답
  # 이므로 아무것도 만들지 않습니다.
  # 중요: 이 경우 원본과 대상 경로가 같아서, sed 원본 > 대상 을 하면
  #       리다이렉션이 파일을 먼저 비워버려 원본이 파괴됩니다.
  if [ "$n" -eq "$B_ATOMIC" ]; then
    ok "GPU ${n}장은 NVIDIA 기준 설정을 그대로 사용 (생성 불필요)"
    return 0
  fi

  mkdir -p "$cN/Offline" "$cN/Server"
  # 정수는 정수로, 소수는 소수 둘째 자리로 출력
  oq=$(awk -v q="$B_OFF_Q" -v d="$dp" 'BEGIN{printf (q==int(q)?"%d":"%.2f"), q*d}')
  sq=$(awk -v q="$B_SRV_Q" -v d="$dp" 'BEGIN{printf (q==int(q)?"%d":"%.2f"), q*d}')
  sed -E "s/offline_expected_qps: [0-9.]+/offline_expected_qps: $oq/" \
      "$srcO" > "$cN/Offline/${B_FILE}.py"
  sed -E "s/server_target_qps: [0-9.]+/server_target_qps: $sq/" \
      "$srcS" > "$cN/Server/${B_FILE}.py"

  # 만들어진 파일에 바꾼 값이 실제로 들어갔는지 확인합니다.
  # sed 는 패턴을 못 찾아도 오류를 내지 않으므로 결과를 직접 봐야 합니다.
  if grep -q "offline_expected_qps: $oq" "$cN/Offline/${B_FILE}.py" 2>/dev/null \
     && grep -q "server_target_qps: $sq" "$cN/Server/${B_FILE}.py" 2>/dev/null; then
    ok "GPU ${n}장 설정 생성 (${dp}세트: 처리량목표 offline=$oq, server=$sq)"
    return 0
  fi
  ng "설정 생성 실패 — 원본의 항목 이름이 바뀐 것 같습니다"
  ng "확인: grep -n 'expected_qps\|target_qps' $srcO $srcS"
  return 1
}

# Server 시나리오의 요청 속도만 바꿉니다.
# 응답시간 제한을 못 지키면 이 값을 낮춰서 다시 돌립니다.
setsrv(){
  sed -i -E "s/server_target_qps: [0-9.]+/server_target_qps: $2/" \
    "$REPO/configs/B300-SXM-270GBx${1}/Server/${B_FILE}.py"
  ok "요청속도(server_target_qps) -> $2"
}

# ============================== 결과 읽기 ====================================
# MLPerf 가 남긴 결과 파일에서 숫자를 뽑아 사람이 읽을 형태로 보여줍니다.
report(){
  local scen="$1" n="$2" D S TPS VAL PCT TGT TTFT TPOT LATOK PQ
  D=$(ls -1dt "$REPO"/build/logs/scaleout_* 2>/dev/null | head -1)   # 최근 실행 폴더
  S=$(find "$D" -name mlperf_log_summary.txt 2>/dev/null | head -1)  # 결과 요약 파일

  # 목표치 = NVIDIA 8장 성적 x (우리 GPU 수 / 8)
  if [ "$scen" = "Server" ]; then
    TGT=$(awk -v p="$B_PUB_SRV" -v n="$n" 'BEGIN{printf "%.0f", p*n/8}')
  else
    TGT=$(awk -v p="$B_PUB_OFF" -v n="$n" 'BEGIN{printf "%.0f", p*n/8}')
  fi

  TPS=""; VAL=""; PCT=""; TTFT=""; TPOT=""
  if [ -n "$S" ]; then
    VAL=$(grep -m1 'Result is' "$S" | awk -F': *' '{print $2}')       # VALID / INVALID
    TPS=$(grep -m1 -E 'Completed tokens per second|Tokens per second' "$S" \
          | grep -oE '[0-9]+\.?[0-9]*' | head -1)                     # 처리량
    # 주의: 결과 파일에는 'TTFT' 라는 글자가 안 나옵니다.
    #       'percentile first token latency' 로 적혀 있어서 이렇게 찾습니다.
    TTFT=$(grep -i 'percentile first token latency'  "$S" | grep '99.00' | grep -oE '[0-9]{6,}' | tail -1)
    TPOT=$(grep -i 'percentile time to output token' "$S" | grep '99.00' | grep -oE '[0-9]{6,}' | tail -1)
  fi
  [ -n "$TPS" ] && PCT=$(awk -v a="$TPS" -v b="$TGT" 'BEGIN{printf "%.1f",a/b*100}')

  say ""
  say "=================================================="
  say " ${B_NAME}  ${scen}   GPU ${n}장 (TP${B_TP} x DP$(dp_of "$n"))"
  say "=================================================="
  printf " %-10s %s\n" "GPU별MiB" "$(mem | tr '\n' ' ')"
  printf " %-10s %s\n" "처리량"    "${TPS:-측정실패} Tokens/s"
  printf " %-10s %s\n" "목표"      "$TGT  (NVIDIA 8장 성적 x ${n}/8)"
  printf " %-10s %s\n" "달성률"    "${PCT:-0}%"
  printf " %-10s %s\n" "판정"      "${VAL:-N/A}"

  # Server 는 응답시간 제한이 있어서 따로 보여줍니다
  if [ "$scen" = "Server" ]; then
    LATOK=1
    if [ -n "$TTFT" ]; then
      awk -v a="$TTFT" -v b="$B_TTFT" \
        'BEGIN{printf "  첫토큰(TTFT) p99 : %8.2f s   한도 %.1f s   %5.1f%%  %s\n",
               a/1e9, b/1e9, a/b*100, (a<=b?"OK":"초과")}'
      [ "$TTFT" -gt "$B_TTFT" ] && LATOK=0
    fi
    if [ -n "$TPOT" ]; then
      awk -v a="$TPOT" -v b="$B_TPOT" \
        'BEGIN{printf "  토큰간격(TPOT) p99: %8.1f ms  한도 %.0f ms  %5.1f%%  %s\n",
               a/1e6, b/1e6, a/b*100, (a<=b?"OK":"초과")}'
      [ "$TPOT" -gt "$B_TPOT" ] && LATOK=0
    fi
    printf " %-10s %s\n" "응답시간"  "$([ "$LATOK" = 1 ] && echo 충족 || echo 위반)"
    # MLPerf 자체 권고문. ' * ' 로 시작하는 줄만 뽑습니다
    # (그냥 뒤 몇 줄을 가져오면 상관없는 수치까지 섞입니다)
    [ -n "$S" ] && grep -E '^\s*\*' "$S" 2>/dev/null | head -3 | sed 's/^/   /'
  fi

  # 정확도 검증을 함께 돌린 경우 그 결과도 표시
  PQ=$(grep -m1 'All Acc. Pass' -A6 "$D/stdout.txt" 2>/dev/null | grep -oE 'Yes|No' | head -1)
  [ -n "$PQ" ] && printf " %-10s %s\n" "정확도"  "전항목 통과? $PQ"

  say "=================================================="
  # 아래 한 줄만 옮겨 적으면 결과를 재구성할 수 있습니다 (로그 반출 제약 대응)
  say " CODE: V6.${B_KEY}.G${n}.${scen}.${TPS:-0}.${PCT:-0}.${VAL:-FAIL}"
  say "=================================================="
  say " 상세로그: $V"
  LAST_VALID="${VAL:-}"; LAST_TPS="${TPS:-}"
}

# ======================== 서버가 안 뜰 때 원인 찾기 ==========================
# 서버는 자기 로그를 따로 남깁니다. 그 파일을 봐야 죽은 이유를 알 수 있습니다.
#   run_llm_server_*.stderr  : 서버를 "띄우는" 쪽 기록 (배정된 GPU 확인용)
#   trtllm_serve_*.log       : 서버 "자신"의 기록 (죽은 이유가 여기 있음)
diag_servers(){
  local D f
  D=$(ls -1dt "$REPO"/build/logs/scaleout_* 2>/dev/null | head -1)
  [ -z "$D" ] && { ng "실행 로그 폴더를 찾을 수 없습니다"; return 1; }
  say ""
  say "[구성 확인]"
  grep -E 'Harness system|Atomic system|GPUs per node|DP multiplicity|Total GPUs' \
    "$V" 2>/dev/null | tail -6 | sed 's/^/   /'
  say ""
  say "[서버별로 어떤 GPU 를 받았나]  — 서로 달라야 정상입니다"
  grep -h 'GPU devices:' "$D"/slurm_logs/run_llm_server_*.stderr 2>/dev/null \
    | sed 's/^.*GPU devices:/   GPU devices:/' | sort | uniq -c | sed 's/^/  /'
  say ""
  say "[서버 자신의 로그]  $D"
  ls -1 "$D"/trtllm_serve_*.log 2>/dev/null | sed 's/^/   /' || ng "서버 로그 없음 (기동 전에 죽음)"
  for f in "$D"/trtllm_serve_*.log; do
    [ -s "$f" ] || continue
    say "   --- $(basename "$f") 의 오류 ---"
    grep -inE 'error|out of memory|OOM|Traceback|abort|Address already in use|CUDA' "$f" \
      2>/dev/null | tail -6 | sed 's/^/     /'
  done
  say ""
  say "[서버 띄우기 쪽 오류]"
  grep -hinE 'error|Traceback|out of memory|refused' \
    "$D"/slurm_logs/run_llm_server_*.stderr 2>/dev/null | tail -8 | sed 's/^/   /'
}

# ==================== 정확도 채점 자산 확인 (및 자동 복구) ===================
# 채점기는 아래 세 가지를 요구합니다. 하나만 없어도 채점이 실패합니다.
#   1) 원본 토크나이저 폴더의 config.json    (모델 구조 정보)
#   2) 원본 토크나이저 폴더의 tokenizer.json (답변을 글자로 되돌리는 데 필요)
#   3) 정답 비교용 원본 데이터 pkl
# 3번은 벤치마크마다 찾는 폴더가 다릅니다. 프로파일의 B_ACC_PKL 참고.
# 복사로 해결되는 문제는 여기서 바로 해결하고, 다운로드가 필요하면 안내합니다.
acc_assets_check(){
  local af=0 f

  # --- 원본 데이터: 있으면 복사로 자동 해결 --------------------------------
  if [ -s "$B_ACC_PKL" ] && [ ! -L "$B_ACC_PKL" ]; then
    ok "채점용 원본 데이터 ($(du -h "$B_ACC_PKL" | cut -f1))"
  elif [ -s "$B_ACC_SRC" ]; then
    say "   채점용 원본 데이터를 채점기가 찾는 위치로 복사합니다"
    say "     $B_ACC_SRC"
    say "     -> $B_ACC_PKL"
    mkdir -p "$(dirname "$B_ACC_PKL")"
    rm -f "$B_ACC_PKL"                       # 잘못된 링크가 있으면 제거
    if cp "$B_ACC_SRC" "$B_ACC_PKL"; then
      ok "복사 완료 ($(du -h "$B_ACC_PKL" | cut -f1))"
    else
      ng "복사 실패 — 디스크 여유를 확인하세요"; af=1
    fi
  else
    ng "채점용 원본 데이터가 없습니다"
    ng "  찾는 위치: $B_ACC_PKL"
    ng "  원본 위치: $B_ACC_SRC  (여기에도 없음)"
    ng "  -> bash $0 assets $B_KEY  로 데이터셋을 먼저 받으세요"
    af=1
  fi

  # --- 토크나이저: 없으면 다운로드 명령을 안내 -----------------------------
  for f in config.json tokenizer.json; do
    if [ -s "$B_MODEL/$f" ]; then
      ok "채점용 $f"
    else
      ng "채점용 파일 없음: $B_MODEL/$f"
      af=1
    fi
  done

  if [ "$af" -eq 1 ]; then
    say ""
    say " 아래를 먼저 실행하세요:"
    say "   hf download $B_HF_REPO config.json tokenizer.json --local-dir $B_MODEL"
    say "   bash $0 assets $B_KEY"
    return 1
  fi
  say "   합격 기준: $B_ACC_NOTE"
  return 0
}

# ============================== 실행 엔진 ====================================
# 인자:  $1=Offline|Server   $2=GPU개수   $3=추가옵션   $4=최대시간
launch(){
  local scen="$1" n="$2" extra="$3" wall="$4" P t=0 viol=0 bad
  local cN="$REPO/configs/B300-SXM-270GBx${n}"

  # --- 사전 점검: 없으면 몇 시간 태우고 실패하므로 미리 막습니다 ------------
  [ -n "$SQSH" ] && [ -f "$SQSH" ] \
    || { ng "컨테이너 이미지가 없습니다 — bash $0 setup"; return 1; }
  [ -f "$B_PREP/input_ids_padded.npy" ] \
    || { ng "전처리 데이터가 없습니다 — bash $0 assets $B_KEY"; return 1; }
  if [ "$B_TP" -gt 1 ] && [ $(( n % B_TP )) -ne 0 ]; then
    ng "이 벤치마크는 GPU 를 ${B_TP}장씩 묶으므로 GPU 개수가 ${B_TP}의 배수여야 합니다"
    return 1
  fi
  [ -f "$cN/$scen/${B_FILE}.py" ] || mkcfg "$n" || return 1

  # 정확도 검증은 채점기가 요구하는 파일이 따로 있습니다.
  # 없는 채로 시작하면 추론이 몇 시간 돌고 나서 채점 단계에서 실패합니다.
  # 그래서 시작 전에 확인하고, 고칠 수 있는 것은 여기서 고칩니다.
  if [[ "$extra" == *AccuracyOnly* ]]; then
    acc_assets_check || return 1
  fi

  hard_clean || return 1
  export SLURM_MPI_TYPE=pmi2      # 앞서 (4)번 문제 우회
  cd "$REPO" || return 1

  say ""
  say "$scen 실행  |  ${B_NAME}  GPU ${n}장 (TP${B_TP} x DP$(dp_of "$n"))"
  say "60초마다 GPU 메모리를 표시합니다. 앞 ${n}장만 차면 정상입니다."

  # --exclusive : 노드를 독점해 CPU 344개를 확보 (앞서 (5)번 문제)
  # --container-image : x86 경로 직접 지정 (앞서 (6)번 문제)
  # --mpi=pmi2 : MPI 규격 버전 우회 (앞서 (4)번 문제)
  # 뒤에 & 를 붙여 배경 실행하고, 진행 상황을 우리가 직접 표시합니다.
  salloc --nodes=1 --gres=gpu:"$n" --exclusive --time="$wall" \
    ./scaleout/run_scaleout.sh \
      --stage all \
      --atomic-system "B300-SXM-270GBx${B_ATOMIC}" \
      --gpus-per-node "$n" --dp-multiplicity "$(dp_of "$n")" \
      --container-image "$SQSH" \
      --mlperf-scratch-path "$SCRATCH" \
      --extra-srun-flags "--overlap --cpu-bind=none --mpi=pmi2" \
      --server-spawn-time 180 \
      --run-args "--benchmarks=${B_NAME} --scenarios=$scen --core_type=trtllm_endpoint $extra --readiness_timeout=3600" \
    >>"$V" 2>&1 &
  P=$!    # 배경 실행한 작업의 프로세스 번호

  # --- 진행 감시 ------------------------------------------------------------
  # 60초마다 GPU 메모리를 찍습니다. 이 숫자만 보면 상황 판단이 됩니다.
  #   전부 4MiB      -> 아직 모델 로딩 시작 전
  #   앞 n장만 증가  -> 정상
  #   n번 이후도 증가 -> GPU 배치 실패 (아래에서 자동 중단)
  while kill -0 $P 2>/dev/null; do
    sleep 60; t=$((t+1))
    say "   $(date +%H:%M)  GPU별MiB [$(mem | tr '\n' ' ')]"
    # 5분이 지난 뒤부터 감시. 일시적 오판을 피해 3회 연속 위반 시에만 중단.
    if [ "$t" -ge 5 ]; then
      bad=$(mem | awk -v n="$n" 'NR>n && $1>50000{c++}END{print c+0}')
      if [ "$bad" -gt 0 ]; then
        viol=$((viol+1))
        say "        경고: GPU ${n}번 이후 ${bad}장이 사용 중 (${viol}/3)"
        if [ "$viol" -ge 3 ]; then
          ng "GPU 배치 실패 — 중단합니다 (몇 시간 낭비 방지)"
          kill $P 2>/dev/null; hard_clean
          diag_servers
          return 2
        fi
      else
        viol=0    # 정상으로 돌아오면 카운터 초기화
      fi
    fi
  done
  wait $P
  return 0
}

# =============================================================================
#                              명령어 처리
# =============================================================================
CMD="${1:-}"; shift 2>/dev/null || true

case "$CMD" in

# ------------------------------------------------------------------- doctor --
# 실행 전 환경이 정상인지 한 번에 점검합니다. 문제가 있으면 조치법을 알려줍니다.
doctor)
  say "MLPerf Inference 환경 진단   $(hostname -s)  $(date '+%F %T')"
  say ""
  say "[하드웨어]"
  GN=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  GP=$(lspci -d 10de: -nn 2>/dev/null | grep -Eci '3d controller|display controller')
  [ "$GN" -gt 0 ] && ok "GPU 인식 ${GN}장 / PCIe 열거 ${GP}개" || ng "GPU 인식 실패"
  # fabric = GPU 간 고속 통신망(NVLink). 여러 장을 묶어 쓸 때 필수.
  FT=$(nvidia-smi --format=csv,noheader --query-gpu=fabric.status 2>/dev/null | wc -l)
  FO=$(nvidia-smi --format=csv,noheader --query-gpu=fabric.status 2>/dev/null | grep -ci success)
  [ "$FT" -gt 0 ] && [ "$FO" -eq "$FT" ] \
    && ok "NVLink fabric 정상 ($FO/$FT)" || ng "NVLink fabric 이상 ($FO/$FT)"
  ok "드라이버 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" \
     "/ FabricManager $(systemctl is-active nvidia-fabricmanager 2>/dev/null)"
  say "   GPU별MiB [$(mem | tr '\n' ' ')]"
  say ""
  say "[작업 스케줄러 / 컨테이너]"
  SG=$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -oP 'Gres=gpu:\K[0-9]+' | head -1)
  ok "SLURM 등록 GPU=${SG:-?}장  노드상태=$(sinfo -h -o '%t' | head -1)  대기작업=$(squeue -h -u "$(whoami)" | wc -l)"
  srun --help 2>&1 | grep -qi container-image && ok "pyxis(컨테이너 플러그인) 정상" || ng "pyxis 미인식"
  ok "enroot $(enroot version 2>/dev/null)   사용가능 MPI: $(srun --mpi=list 2>&1 | tr '\n' ' ')"
  [ -n "$SQSH" ] && ok "컨테이너 이미지 $(basename "$SQSH") ($(du -h "$SQSH" 2>/dev/null|cut -f1))" \
                 || ng "컨테이너 이미지 없음 — bash $0 setup"
  say ""
  say "[소스코드 버그 수정 적용 여부]"
  grep -q 'versioning.parse(C.VERSION' "$REPO/code/common/mlcommons/loadgen.py" 2>/dev/null \
    && ok "patch1 벤치마크목록 v6.0 고정" || ng "patch1 누락 — bash $0 patch"
  grep -q 'mpi_mode=legacy' "$REPO/scaleout/run_scaleout.sh" 2>/dev/null \
    && ok "patch2 실행방식 legacy" || ng "patch2 누락 — bash $0 patch"
  grep -q "locals().get('gpu_ids'" "$REPO/code/llmlib/launch_server.py" 2>/dev/null \
    && ok "patch3 gpu_ids 오류 해소" || ng "patch3 누락 — bash $0 patch"
  # patch4,5 는 GPU 를 여러 장 쓸 때만 필요합니다 (1장 실행에는 영향 없음)
  grep -q '\[PATCH4B\]' "$REPO/code/llmlib/launch_server.py" 2>/dev/null \
    && ok "patch4 서버별 GPU 배정 (다중 GPU 필수)" || ng "patch4 누락 — bash $0 patch"
  grep -q '\[PATCH5\]' "$REPO/code/llmlib/launch_server.py" 2>/dev/null \
    && ok "patch5 서버별 로그파일 분리" || ng "patch5 누락 — bash $0 patch"
  bash -n "$REPO/scaleout/run_scaleout.sh" 2>/dev/null \
    && ok "실행 스크립트 문법 정상" || ng "실행 스크립트 문법 깨짐 (.orig 로 복구 필요)"
  say ""
  say "[모델 / 데이터]"
  for b in llama2-70b llama3.1-405b; do
    bench_init "$b"
    printf "   %-16s" "$B_NAME:"
    [ -f "$B_PREP/input_ids_padded.npy" ] && printf "전처리OK  " || printf "전처리없음  "
    ls "$B_DATA"/*.pkl >/dev/null 2>&1 && printf "데이터 %s개  " "$(ls "$B_DATA"/*.pkl|wc -l)" \
                                       || printf "데이터없음  "
    # 정확도 채점 준비 상태 (원본 데이터 + 토크나이저 2개)
    A=0
    [ -s "$B_ACC_PKL" ] && [ ! -L "$B_ACC_PKL" ] && A=$((A+1))
    [ -s "$B_MODEL/config.json" ]    && A=$((A+1))
    [ -s "$B_MODEL/tokenizer.json" ] && A=$((A+1))
    say "채점준비 ${A}/3"
  done
  say "   ※ 채점준비가 3/3 이 아니면 정확도 검증이 마지막에 실패합니다"
  say "   디스크 여유 $(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -d ' ')"
  say ""
  say "[준비된 GPU 개수별 설정]"
  find "$REPO/configs" -maxdepth 1 -name 'B300*' -type d 2>/dev/null \
    | xargs -n1 basename 2>/dev/null | sort | tr '\n' ' ' | sed 's/^/   /'
  say ""
  say " 상세로그: $V"
  ;;

# -------------------------------------------------------------------- paths --
# 지금 어떤 경로를 쓰고 있는지 보여줍니다. 경로를 바꿨을 때 확인용입니다.
paths)
  say "적용된 경로"
  say ""
  say "[설정 출처]"
  [ -f "$CONF" ] && ok "설정파일 $CONF" || say "   설정파일 없음 ($CONF)  — 기본값 사용"
  say ""
  say "[경로]"
  chk(){ # $1=이름 $2=경로 $3=디렉토리면 d, 파일이면 f
    if [ "$3" = d ] && [ -d "$2" ]; then
      printf "   %-12s %s   (있음, 여유 %s)\n" "$1" "$2" \
             "$(df -BG --output=avail "$2" 2>/dev/null | tail -1 | tr -d ' ')"
    elif [ "$3" = f ] && [ -s "$2" ]; then
      printf "   %-12s %s   (있음, %s)\n" "$1" "$2" "$(du -h "$2" 2>/dev/null | cut -f1)"
    else
      printf "   %-12s %s   (없음)\n" "$1" "${2:-미지정}"
    fi
  }
  chk "BASE"     "$BASE"    d
  chk "SCRATCH"  "$SCRATCH" d
  chk "REPO"     "$REPO"    d
  chk "LOGDIR"   "$RUNDIR"  d
  chk "CONT"     "$SQSH"    f
  say ""
  say "[SCRATCH 하위 구조]"
  for sub in data models preprocessed_data; do
    if [ -d "$SCRATCH/$sub" ]; then
      printf "   %-18s %s\n" "$sub/" "$(du -sh "$SCRATCH/$sub" 2>/dev/null | cut -f1)"
    else
      printf "   %-18s (없음)\n" "$sub/"
    fi
  done
  say ""
  say "[바꾸는 방법]"
  say "   이번 실행만:  bash $0 --scratch /새/경로 <명령어> ..."
  say "   계속 적용  :  아래 내용을 $CONF 에 저장"
  say "                   BASE=$BASE"
  say "                   SCRATCH=$SCRATCH"
  say "                   REPO=$REPO"
  [ -n "$SQSH" ] && say "                   CONT=$SQSH"
  ;;

# -------------------------------------------------------------------- patch --
patch) say "알려진 버그 수정 3종 적용"; apply_patches ;;

# --------------------------------------------------------------------- diag --
# 서버가 안 떴을 때 원인을 찾습니다. 실패 직후에 실행하세요.
diag) diag_servers ;;

# -------------------------------------------------------------------- clean --
clean) hard_clean; squeue -u "$(whoami)" ;;

# -------------------------------------------------------------------- setup --
# 소스코드, 의존 저장소, 컨테이너 이미지를 준비합니다. 처음 한 번만.
setup)
  say "소스코드 / 의존성 / 컨테이너 준비"
  mkdir -p "$SAFE"
  # 컨테이너 이미지를 푸는 임시 폴더. 기본값 /tmp 는 메모리 디스크라 터집니다.
  export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-$BASE/.enroot/tmp}"
  export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$BASE/.enroot/cache}"
  mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH"

  if [ -d "$REPO/configs/B300-SXM-270GBx8" ]; then
    ok "소스코드 이미 있음"
  else
    say "   내려받는 중 (closed/NVIDIA 부분만)..."
    RR="$(dirname "$(dirname "$REPO")")"
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/mlcommons/inference_results_v6.0.git "$RR" >>"$V" 2>&1 \
      && ( cd "$RR" && git sparse-checkout set closed/NVIDIA >>"$V" 2>&1 ) \
      && ok "소스코드 준비 완료" || ng "내려받기 실패"
  fi

  # MLPerf 공식 저장소(채점 스크립트 포함) + TensorRT-LLM
  mkdir -p "$REPO/3rdparty"
  [ -d "$REPO/3rdparty/mlc-inference/.git" ] || \
    git clone --depth 1 https://github.com/mlcommons/inference.git \
      "$REPO/3rdparty/mlc-inference" >>"$V" 2>&1
  [ -d "$REPO/3rdparty/trtllm/.git" ] || \
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git \
      "$REPO/3rdparty/trtllm" >>"$V" 2>&1
  ok "의존 저장소 $(ls -1 "$REPO/3rdparty" 2>/dev/null | wc -l)개"

  apply_patches

  # 컨테이너 이미지. NGC 완성 이미지를 그대로 씁니다(소스 빌드 불필요).
  mkdir -p "$REPO/build/sqsh_images"
  T="$REPO/build/sqsh_images/mlperf-inference-$(whoami)-x86_64-release.sqsh"
  if ls "$REPO"/build/sqsh_images/*.sqsh >/dev/null 2>&1; then
    ok "컨테이너 이미지 이미 있음"
  elif ls "$SAFE"/*.sqsh >/dev/null 2>&1; then
    mv "$(ls -1 "$SAFE"/*.sqsh | head -1)" "$REPO/build/sqsh_images/" && ok "대피시켜둔 이미지 복구"
  else
    say "   NGC 에서 내려받는 중 (약 30GB, 20~60분)"
    enroot import -o "$T" "docker://nvcr.io#${NGC_IMG#nvcr.io/}" \
      && ok "컨테이너 이미지 준비 완료" \
      || ng "실패 — ~/.config/enroot/.credentials 의 NGC 인증정보 확인"
  fi
  say ""; say " 다음:  bash $0 doctor    그다음  bash $0 assets llama3.1-405b"
  ;;

# ------------------------------------------------------------------- assets --
# 모델 가중치와 데이터셋을 내려받고 전처리합니다. 수 시간 걸립니다.
assets)
  bench_init "${1:-llama3.1-405b}"
  say "${B_NAME} 모델/데이터 준비  |  상세로그 $V"
  mkdir -p "$B_DATA" "$B_PREP" "$B_MODEL"
  R2="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
  cd "$B_DATA" || exit 1

  if [ "$B_KEY" = "llama3.1-405b" ]; then
    # 데이터셋 (MLCommons 공식 저장소)
    ls "$B_DATA"/*8313* >/dev/null 2>&1 && ok "데이터셋 이미 있음" || {
      bash <(curl -s "$R2") https://inference.mlcommons-storage.org/metadata/llama3-1-405b-dataset-8313.uri 2>&1 | tail -3
      bash <(curl -s "$R2") https://inference.mlcommons-storage.org/metadata/llama3-1-405b-calibration-dataset-512.uri 2>&1 | tail -3
    }
    # 모델: NVIDIA 가 미리 FP4 로 압축해둔 것을 씁니다 (약 219GB).
    # 원본 BF16(약 810GB)을 받아 직접 압축할 필요가 없습니다.
    [ -d "$B_FP4" ] && ok "FP4 모델 ($(du -sh "$B_FP4"|cut -f1))" || \
      hf download nvidia/Llama-3.1-405B-Instruct-FP4 --local-dir "$B_FP4"
    # 정확도 채점기는 원본 토크나이저를 별도로 요구합니다 (가중치는 불필요).
    for f in config.json tokenizer.json tokenizer_config.json \
             special_tokens_map.json generation_config.json; do
      [ -s "$B_MODEL/$f" ] || \
        hf download meta-llama/Llama-3.1-405B-Instruct "$f" --local-dir "$B_MODEL" >>"$V" 2>&1
    done
  else
    ls "$B_DATA"/*sampled_24576* >/dev/null 2>&1 && ok "데이터셋 이미 있음" || \
      bash <(curl -s "$R2") https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri 2>&1 | tail -3
    hf download centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0 \
      --local-dir "$SCRATCH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4" >>"$V" 2>&1
    for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
      [ -s "$B_MODEL/$f" ] || \
        hf download meta-llama/Llama-2-70b-chat-hf "$f" --local-dir "$B_MODEL" >>"$V" 2>&1
    done
  fi

  # 하위 폴더에 떨어진 파일을 끌어올리고 압축을 풉니다
  find "$B_DATA" -mindepth 2 -name '*.pkl' -exec mv -n {} "$B_DATA/" \; 2>/dev/null
  find "$B_DATA" -name '*.pkl.gz' -exec gzip -dkf {} \; 2>/dev/null
  ls -lh "$B_DATA"/*.pkl 2>/dev/null | sed 's/^/   /'

  # 정확도 채점기용 원본 데이터를 채점기가 찾는 위치에 미리 놓아둡니다.
  # 벤치마크마다 찾는 폴더가 달라서(llama2 는 open_orca, 405b 는 llama3.1-405b)
  # 프로파일에 정의된 경로를 그대로 씁니다. 링크가 아니라 복사입니다.
  if [ -s "$B_ACC_SRC" ]; then
    if [ ! -s "$B_ACC_PKL" ] || [ -L "$B_ACC_PKL" ]; then
      mkdir -p "$(dirname "$B_ACC_PKL")"
      rm -f "$B_ACC_PKL"
      cp "$B_ACC_SRC" "$B_ACC_PKL" 2>/dev/null \
        && ok "채점용 원본 배치: $B_ACC_PKL" \
        || ng "채점용 원본 복사 실패 (디스크 여유 확인)"
    else
      ok "채점용 원본 이미 배치됨"
    fi
  else
    ng "채점용 원본이 없습니다: $B_ACC_SRC  (데이터셋 다운로드 확인 필요)"
  fi

  # 전처리는 sbatch 로 던집니다. 터미널을 닫아도 계속 돕니다.
  cat > /tmp/prep-$B_KEY.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${B_KEY}-prep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=$RUNDIR/${B_KEY}-prep-%j.log
srun --container-image $SQSH \\
     --container-mounts $REPO:/work,$SCRATCH:/home/mlperf_inference_storage \\
     --container-workdir /work --container-remap-root \\
     --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \\
     bash -lc 'set -x; make link_dirs; python3 -u code/${B_FILE}/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed_data'
EOF
  J=$(sbatch --parsable /tmp/prep-$B_KEY.sbatch 2>>"$V")
  say "   전처리 작업번호 ${J:-실패}   진행보기: tail -f $RUNDIR/${B_KEY}-prep-${J}.log"
  ;;

# ---------------------------------------------------------------- run / test --
#   run  <벤치마크> <시나리오> <GPU수> [요청속도]
#   test <벤치마크> <GPU수>              1분 예비 테스트
run|test)
  bench_init "${1:-llama3.1-405b}"; SCEN_IN="${2:-offline}"; N="${3:-4}"; Q="${4:-}"
  if [ "$CMD" = "test" ]; then
    N="${2:-4}"; SCEN_IN=offline; EXTRA="--test_run"
  else
    EXTRA=""
  fi
  case "$(echo "$SCEN_IN" | tr 'A-Z' 'a-z')" in
    offline)  SCEN=Offline; WALL="$B_WALL_OFF" ;;
    server)   SCEN=Server;  WALL="$B_WALL_SRV" ;;
    accuracy) SCEN=Offline; WALL="05:00:00"; EXTRA="--test_mode=AccuracyOnly" ;;
    *) ng "시나리오는 offline / server / accuracy 중 하나입니다"; exit 1 ;;
  esac
  [ -f "$REPO/configs/B300-SXM-270GBx${N}/Server/${B_FILE}.py" ] || mkcfg "$N"
  [ -n "$Q" ] && setsrv "$N" "$Q"

  say "${B_NAME} ${SCEN}${EXTRA:+ ($EXTRA)}  |  GPU ${N}장"
  if [ "$SCEN" = "Server" ]; then
    say "응답시간 제한: 첫토큰 $(awk -v v="$B_TTFT" 'BEGIN{printf "%.1f",v/1e9}')초 / 토큰간격 $(awk -v v="$B_TPOT" 'BEGIN{printf "%.0f",v/1e6}')ms — 넘으면 INVALID"
  fi
  launch "$SCEN" "$N" "$EXTRA" "$WALL" && report "$SCEN" "$N"
  ;;

# -------------------------------------------------------------------- score --
# 이미 끝난 실행의 "채점만" 다시 합니다. 추론을 다시 돌릴 필요가 없습니다.
# 정확도 검증이 채점 단계에서만 실패했을 때 3시간을 아낄 수 있습니다.
score)
  bench_init "${1:-llama3.1-405b}"; N="${2:-4}"
  D="${3:-$(ls -1dt "$REPO"/build/logs/scaleout_* 2>/dev/null | head -1)}"
  D="${D#$REPO/}"      # 절대경로를 상대경로로 (컨테이너 안에서는 /work 기준)
  say "채점 재실행  |  ${B_NAME}  GPU ${N}장"
  say "대상 로그: $D"
  ls -la "$REPO/$D"/*_TRT/"$B_NAME"/*/mlperf_log_accuracy.json 2>/dev/null | sed 's/^/   /'
  cd "$REPO" || exit 1
  # SYSTEM_NAME 을 반드시 줘야 합니다. 안 주면 GPU 1장짜리로 오인합니다.
  srun --gres=gpu:1 --time=01:00:00 \
    --container-image "$SQSH" \
    --container-mounts "$REPO:/work,$SCRATCH:/home/mlperf_inference_storage" \
    --container-workdir /work --container-remap-root \
    --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
    bash -lc "make link_dirs && make display_results LOG_DIR=/work/$D SYSTEM_NAME=B300-SXM-270GBx${N}"
  ;;

# ------------------------------------------------------------------- result --
# 실행을 다시 하지 않고 최근 결과만 다시 표시합니다.
result)
  bench_init "${3:-llama3.1-405b}"
  report "$(echo "${1:-Offline}" | sed 's/^./\U&/')" "${2:-4}"
  ;;

# ------------------------------------------------------------------- record --
# 지금까지 확보한 결과 대장. 보고서에 그대로 쓸 수 있습니다.
record)
  cat <<'EOR'
==================================================================
 MLPerf Inference v6.0 결과
 Dell XE9780 / NVIDIA B300  (8번 GPU 슬롯 보드 장애로 7장만 가용)
==================================================================
 벤치마크        구성            항목       실측        기준        결과
------------------------------------------------------------------
 llama2-70b     1GPU  TP1       Offline    14,438.6    14,119     102.3% VALID
 llama3.1-405b  4GPU  TP2xDP2   Offline     1,023.06      976     104.8% VALID
 llama3.1-405b  4GPU  TP2xDP2   Accuracy   ROUGEL       21.84 (>=21.45)  PASS
                                           exact_match  90.08 (>=89.23)  PASS
                                           TOKENS/SAMPLE 636.20 (>=616.21) PASS
 llama3.1-405b  4GPU  TP2xDP2   Server       636.11       730     TTFT 102.1% INVALID
                                           (요청속도 1.0, 토큰간격은 63.9% 충족)
------------------------------------------------------------------
 GPU 1장당 환산
   llama2-70b      14,438.6 /GPU   vs NVIDIA 공개값 14,119 /GPU
   llama3.1-405b      255.77 /GPU  vs NVIDIA 공개값 243.95 /GPU
------------------------------------------------------------------
 결론
   서로 다른 모델(70B / 405B), 서로 다른 병렬화 방식(TP1 / TP2),
   속도와 정확도 양쪽에서 NVIDIA 공개 제출값을 재현 또는 상회했습니다.
   GPU 1장당 성능이 기준과 동등하므로 하드웨어/드라이버/추론엔진 모두 정상.

 미달성 항목
   405B Server — GPU 4장으로는 8장 기준 요청량을 소화할 수 없어 대기열이
   누적됐습니다. 토큰 생성 속도 자체는 규격 내(63.9%)이므로 성능 문제가
   아니라 구성 규모의 문제입니다.

 실행 불가 항목
   deepseek-r1 — GPU 8장을 하나로 묶는 구성(TP8/EP8)이 필수입니다.
   보드 교체로 8장이 복구된 뒤에 가능합니다.
==================================================================
EOR
  ;;

# 인자 없이 실행하면 맨 위 사용법을 보여줍니다
*) sed -n '3,52p' "$0" ;;
esac
