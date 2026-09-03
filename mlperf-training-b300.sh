#!/usr/bin/env bash
# =============================================================================
#  MLPerf Training v6.0 실행기  —  Llama 3.1 8B (NVIDIA NeMo)
#  Dell PowerEdge XE9780 / NVIDIA B300   1노드 x 4 GPU   NVFP4
#
#  이 스크립트가 하는 일
#    MLPerf Training 은 "모델을 목표 품질까지 학습시키는 데 걸린 시간"을 재는
#    벤치마크입니다. 추론(Inference)과는 완전히 다른 벤치마크이고 저장소도
#    다릅니다. 추론은 mlperf-inference-b300.sh 를 쓰세요.
#
#    여기서는 Llama 3.1 8B 모델을 GPU 4장으로 학습시킵니다.
#    실제 수렴까지는 수십~수백 시간이 걸리므로, 보통은 짧게 돌려
#    "1스텝당 몇 초 걸리나 = 얼마나 빠른가"만 측정합니다.
#
# -----------------------------------------------------------------------------
#  사용법
# -----------------------------------------------------------------------------
#  [1단계] 준비
#    bash mlperf-training-b300.sh check     자산/설정/환경 점검
#    bash mlperf-training-b300.sh doctor    실행환경 상세 진단 + 최근 실패 원인
#    bash mlperf-training-b300.sh build     컨테이너 이미지 만들기
#    bash mlperf-training-b300.sh prep      학습 데이터 내려받기 (약 85GB)
#    bash mlperf-training-b300.sh inspect   컨테이너 내부 라이브러리 점검
#
#  [2단계] 실행 — 위에서 아래 순서로
#    bash mlperf-training-b300.sh dryrun       설정만 출력, 학습은 안 함
#    bash mlperf-training-b300.sh smoke [50]   가짜 데이터로 동작만 확인
#    bash mlperf-training-b300.sh perf  [200]  실제 데이터로 속도 측정  <-- 본론
#    bash mlperf-training-b300.sh fp8attn [200] FP8 어텐션 켜고 비교
#    bash mlperf-training-b300.sh sweep [150]  배치크기 1/2/4 자동 비교
#    bash mlperf-training-b300.sh converge     실제 수렴까지 (매우 오래 걸림)
#    bash mlperf-training-b300.sh nccl         GPU 간 통신속도만 측정
#
#  [3단계] 결과 / 정리
#    bash mlperf-training-b300.sh result    최근 결과 다시 보기
#    bash mlperf-training-b300.sh clean     멈춘 작업/컨테이너/GPU 강제 정리
#
# -----------------------------------------------------------------------------
#  용어
# -----------------------------------------------------------------------------
#  step        학습 1회 반복. 데이터 한 묶음을 보고 모델을 조금 고치는 단위.
#  GBS         Global Batch Size. 1 step 에서 보는 전체 데이터 양.
#  MBS         Micro Batch Size. GPU 가 한 번에 처리하는 양.
#  GA          Gradient Accumulation. MBS 를 몇 번 모아 1 step 을 만드는지.
#              GBS = MBS x GA x (병렬 세트 수) 관계입니다.
#  TP/PP/CP    모델을 GPU 에 쪼개 올리는 세 가지 방식.
#              이 설정은 전부 1 이라 쪼개지 않고 4장에 각각 통째로 올립니다.
#  DP          Data Parallel. 그렇게 만든 세트가 몇 개인지. 여기서는 4.
#  TFLOPS/GPU  GPU 1장이 초당 몇 조 번 연산했는지. 성능의 핵심 지표.
#  log_ppl     학습 품질 지표. 낮을수록 좋고, 3.3 아래로 내려가면 목표 달성.
#  synthetic   가짜 데이터. 동작 확인용이며 성능 수치로 쓰면 안 됩니다.
#
# -----------------------------------------------------------------------------
#  성능은 어떻게 계산하나
# -----------------------------------------------------------------------------
#  학습 로그(mllog)에 매 step 소요 시간이 train_step_time 으로 기록됩니다.
#  거기서 중앙값을 뽑아 아래처럼 환산합니다.
#
#    TFLOPS/GPU = 421.59(모델 1샘플당 연산량) x GBS / step시간 / GPU수
#    tokens/s   = GBS x 8192(문장 길이) / step시간
#
#  중앙값을 쓰는 이유: 초반 몇 step 은 워밍업으로 느려서 평균을 왜곡합니다.
#  이 스크립트는 앞부분 일부를 자동으로 제외합니다.
#
# -----------------------------------------------------------------------------
#  이 장비에서 실제로 겪은 문제 — 모두 아래 코드에 반영돼 있습니다
# -----------------------------------------------------------------------------
#  (1) pyxis(컨테이너 플러그인)는 로컬 docker 이미지를 읽지 못합니다.
#      CONT 는 반드시 .sqsh 파일의 절대경로여야 합니다.
#      이름만 적으면 Docker Hub 로 찾아가서 401 인증오류가 납니다.
#  (2) 컨테이너 안의 TransformerEngine 이 cuDNN 을 못 찾습니다.
#      이미지에 런타임 패키지만 있어 libcudnn.so 링크가 없고, 위치도
#      /usr/local/cuda 가 아니라 /usr/lib/x86_64-linux-gnu 입니다.
#      CUDNN_PATH 를 직접 지정해야 import 가 통과합니다.
#  (3) 컨테이너의 OpenMPI 가 PMIx 를 요구하지만 이 SLURM 에는 없습니다.
#      학습 자체는 SLURM 랭크 정보를 직접 쓰므로 무관하지만, 통신 테스트는
#      실패하므로 기본으로 끕니다(NCCL_TEST=0).
#  (4) sbatch 는 제출한 쉘의 환경변수를 그대로 컨테이너에 넘깁니다.
#      호스트의 LD_LIBRARY_PATH 가 컨테이너 안 라이브러리를 가리면 치명적이라
#      제출 직전에 자동으로 제거합니다.
#  (5) MAX_STEPS 는 설정파일을 읽은 "뒤에" 덮어써야 합니다.
#      설정파일 안에서 바꾸면 학습률 감쇠 계획(OPT_LR_DECAY_STEPS)까지 따라
#      변해서 학습 궤적이 왜곡됩니다.
#  (6) 성공/실패 판정을 SLURM 상태로 하면 안 됩니다. 통신 테스트 단계가
#      실패해도 학습은 정상일 수 있습니다. 학습 로그에 step 기록이 있는지로
#      판단합니다.
# =============================================================================
set -u    # 정의하지 않은 변수를 쓰면 즉시 중단

# ============================== 사용자 설정 ==================================
BASE="${BASE:-/data/lsh}"
# MLPerf Training 소스코드 (Dell 제출본의 NeMo 구현)
REPO="${REPO:-$BASE/training_results_v6.0/Dell/benchmarks/llama31_8b/implementations/nemo}"
DATADIR="${DATADIR:-$BASE/mlperf_training_data}"   # 학습 데이터 (약 90GB)
LOGDIR="${LOGDIR:-$BASE/mlperf-8b-logs}"           # 학습 로그
CONT="${CONT:-$BASE/llama31_8b.sqsh}"              # 컨테이너 이미지 (절대경로!)
DOCKER_TAG="${DOCKER_TAG:-mlperf-nvidia:llama31_8b-pyt}"   # docker build 태그

# 설정파일 두 개. 기본과 FP8 어텐션 변형을 A/B 비교할 수 있습니다.
CFG_BASE="config_XE9780_B300_1x4x4xtp1pp1cp1_8b_fp4.sh"
CFG_ATTN="config_XE9780_B300_1x4x4xtp1pp1cp1_8b_fp4_fp8attn.sh"

# cuDNN 위치. 앞서 (2)번 문제 때문에 직접 지정합니다.
# TransformerEngine 의 탐색 순서는
#   CUDNN_HOME -> CUDNN_PATH -> CUDA_HOME -> CUDA_PATH -> /usr/local/cuda
CUDNN_PATH_DEFAULT="${CUDNN_PATH_DEFAULT:-/usr/lib/x86_64-linux-gnu}"

NGPU=4                  # 사용할 GPU 수
SEQLEN=8192             # 문장 길이 (토큰 수)
TFLOP_PER_SAMPLE=421.59 # 모델 1샘플 학습에 필요한 연산량 (MLPerf 규정값)

MODE="${1:-}"; A2="${2:-}"
V="$BASE/training-$(date +%m%d-%H%M).log"    # 상세 로그
mkdir -p "$LOGDIR" 2>/dev/null

# ============================== 출력 도구 ====================================
say(){ printf '%s\n' "$*"; }
ok(){  printf '   OK   %s\n' "$*"; }
ng(){  printf '   --   %s\n' "$*"; }
hr(){  say "--------------------------------------------------"; }
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; }

# ============================== 강제 정리 ====================================
# 이전 학습이 비정상 종료하면 GPU 메모리와 컨테이너가 남습니다.
# 그 상태로 새로 시작하면 메모리 부족으로 또 실패하므로 확실히 비웁니다.
hard_clean(){
  scancel -u "$(whoami)" >>"$V" 2>&1; sleep 3       # 내 SLURM 작업 전부 취소
  local i pids p
  for i in 1 2 3 4 5 6; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
  # 학습은 컨테이너를 쓰고 버리므로, 남은 컨테이너도 정리합니다
  if command -v enroot >/dev/null 2>&1; then
    [ -n "$(enroot list 2>/dev/null)" ] && enroot remove -f $(enroot list) >>"$V" 2>&1
  fi
  for i in 1 2 3 4 5 6; do        # 작업이 정리중(CG) 상태에서 빠질 때까지
    squeue -h -u "$(whoami)" 2>/dev/null | grep -q . || break
    sleep 5
  done
  local busy; busy=$(mem | awk '$1>1000{n++}END{print n+0}')
  if [ "${busy:-0}" -gt 0 ]; then
    ng "GPU 메모리가 아직 안 비었습니다 [$(mem | tr '\n' ' ')]"
    ng "확인: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    return 1
  fi
  ok "GPU 전부 비움 [$(mem | tr '\n' ' ')]   대기작업 $(squeue -h -u "$(whoami)" 2>/dev/null | wc -l)개"
  return 0
}

# ========================= 설정파일 읽고 해석하기 ============================
# 설정파일은 쉘 스크립트라 source 로 읽으면 변수가 됩니다.
# 그 값들로 병렬 구조와 배치 크기를 계산합니다.
load_cfg(){
  local c="$REPO/$1"
  [ -f "$c" ] || { ng "설정파일이 없습니다: $c"; return 1; }
  set -a; source "$c"; set +a       # set -a: 읽은 변수를 자동으로 export
  CFG="$1"
  MP=$(( TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL ))
  WS=$(( DGXNNODES * DGXNGPU ))     # 전체 GPU 수
  DP=$(( WS / MP ))                 # 병렬 세트 수
  GBS=$(( MINIBS * DP ))            # 1 step 에서 보는 전체 데이터
  GA=$(( MINIBS / MICRO_BATCH_SIZE ))
  # MLPerf 규정: 평가 주기는 12288 샘플마다. GBS 로 나눠 step 수로 환산.
  VCI_AUTO=$(( (12288 + GBS - 1) / GBS ))
  export CFG MP WS DP GBS GA VCI_AUTO
}

# 해석한 설정을 사람이 읽을 수 있게 표시하고, 규정 위반이 있으면 경고합니다.
show_cfg(){
  printf " %-22s %s\n" "설정파일"    "$CFG"
  printf " %-22s %s\n" "시스템"      "$DGXSYSTEM"
  printf " %-22s %s\n" "구성"        "${DGXNNODES}노드 x ${DGXNGPU}GPU = 전체 ${WS}장"
  printf " %-22s %s\n" "병렬화"      "TP${TENSOR_MODEL_PARALLEL} PP${PIPELINE_MODEL_PARALLEL} CP${CONTEXT_PARALLEL} -> ${DP}세트"
  printf " %-22s %s\n" "배치"        "MINIBS ${MINIBS} x ${DP}세트 = GBS ${GBS}   (MBS ${MICRO_BATCH_SIZE}, GA ${GA})"
  printf " %-22s %s\n" "정밀도"      "FP4=${FP4} recipe=${FP4_RECIPE} FP8_DPA=${FP8_DPA:-False} CUDA_GRAPH=${MCORE_CUDA_GRAPH:-0}"
  printf " %-22s %s\n" "학습률"      "LR ${LR}  워밍업 ${WARMUP_STEPS}  감쇠 ${OPT_LR_DECAY_STEPS}"
  printf " %-22s %s\n" "평가주기"    "${VAL_CHECK_INTERVAL} step (규정값 ${VCI_AUTO})  샘플 ${VAL_SAMPLES}"
  printf " %-22s %s\n" "최대 step"   "${MAX_STEPS}"
  # 규정 위반 경고
  [ "$VAL_CHECK_INTERVAL" -ne "$VCI_AUTO" ] && \
    ng "평가주기(${VAL_CHECK_INTERVAL})가 규정값 ceil(12288/GBS)=${VCI_AUTO} 와 다릅니다"
  [ $(( MINIBS % MICRO_BATCH_SIZE )) -ne 0 ] && \
    ng "MINIBS 가 MICRO_BATCH_SIZE 로 안 나누어떨어집니다 (GA 가 정수가 아님)"
  return 0
}

# ========================= 학습 로그 분석기 ==================================
# 학습 로그는 MLPerf 표준 형식(:::MLLOG 로 시작하는 JSON)입니다.
# 아래 파이썬 코드가 그 로그에서 성능 지표를 뽑아냅니다.
PARSE_PY='
import json,sys,statistics as st
path,ngpu,tflop,seq = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
steps=[]; evals=[]; gbs=None; ga=None; t0=t1=None; status=None
for line in open(path, errors="ignore"):
    i=line.find(":::MLLOG")
    if i<0: continue
    try: d=json.loads(line[i+8:])
    except Exception: continue
    k=d.get("key"); v=d.get("value"); m=d.get("metadata") or {}
    # 매 step 소요 시간 — 성능 계산의 원천 데이터
    if k=="tracked_stats" and isinstance(v,dict) and "train_step_time" in v:
        t=v["train_step_time"]
        if t and t>0: steps.append(t)
    elif k=="eval_accuracy": evals.append((m.get("samples_count"), v))
    elif k=="global_batch_size": gbs=v
    elif k=="gradient_accumulation_steps": ga=v
    elif k=="run_start": t0=d.get("time_ms")
    elif k=="run_stop":  t1=d.get("time_ms"); status=m.get("status")
# 초반 워밍업 step 은 느려서 통계를 왜곡합니다. 앞 20%(최대 10개)를 뺍니다.
warm=min(10,max(0,len(steps)//5))
body=steps[warm:] or steps
out={"n_step":len(steps),"n_used":len(body),"n_warm":warm,"gbs":gbs,"ga":ga,
     "status":status,"evals":evals[-5:],
     "walltime_s":(t1-t0)/1000.0 if (t0 and t1) else None}
if body:
    med=st.median(body); srt=sorted(body)
    out.update({
      "step_median":med,"step_mean":sum(body)/len(body),
      "step_min":srt[0],"step_max":srt[-1],
      "step_p90":srt[int(len(srt)*0.9)-1] if len(srt)>=10 else srt[-1],
      "spread_pct":(srt[-1]-srt[0])/med*100.0,
      "tflops_per_gpu":(tflop*gbs/med/ngpu) if gbs else None,
      "tokens_per_s":(gbs*seq/med) if gbs else None,
      "samples_per_s":(gbs/med) if gbs else None,
    })
print(json.dumps(out))
'

# 학습이 실제로 돌았는지 판정. step 기록이 하나라도 있으면 성공으로 봅니다.
# (앞서 (6)번 문제 — SLURM 상태로 판단하면 오판합니다)
has_steps(){
  python3 -c "$PARSE_PY" "$1" "$NGPU" "$TFLOP_PER_SAMPLE" "$SEQLEN" 2>/dev/null \
  | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['n_step']>0 else 1)"
}

# 이번 실행의 로그 파일을 찾습니다. DATESTAMP 로 정확히 특정합니다.
find_mllog(){
  local ds="${1:-}"
  if [ -n "$ds" ]; then
    ls -1t "$LOGDIR"/*"${ds}"*.log 2>/dev/null \
      | grep -v mountcheck | grep -E '_[0-9]+\.log$' | head -1
  else
    ls -1t "$LOGDIR"/*.log 2>/dev/null \
      | grep -v mountcheck | grep -E '_[0-9]+\.log$' | head -1
  fi
}

# ============================== 결과 표시 ====================================
report(){
  local L="${1:-}" J SYN=""
  [ -z "$L" ] && L=$(find_mllog)
  if [ -z "$L" ] || [ ! -f "$L" ]; then
    ng "분석할 학습 로그가 없습니다 ($LOGDIR)"; return 1
  fi
  case "$L" in *_synth_*) SYN=1 ;; esac    # 가짜 데이터 여부

  J=$(python3 -c "$PARSE_PY" "$L" "$NGPU" "$TFLOP_PER_SAMPLE" "$SEQLEN" 2>/dev/null)
  say ""
  say "=================================================="
  say " llama31_8b  ${NGPU}GPU (TP1 PP1 CP1 x ${DP:-4}세트)  GBS ${GBS:-?}"
  say "=================================================="
  printf " %-14s %s\n" "로그파일" "$(basename "$L")"
  if [ -z "$J" ]; then ng "로그 분석 실패"; return 1; fi

  python3 - "$J" "${SYN:-0}" <<'PYEOF'
import json,sys
d=json.loads(sys.argv[1]); syn = sys.argv[2]=="1"
f=lambda k,fmt="%.4f",dflt="N/A": (fmt%d[k]) if d.get(k) is not None else dflt
print(" %-14s %s" % ("측정 step",
      "%d / %d  (앞 %d step 워밍업 제외)" % (d["n_used"],d["n_step"],d["n_warm"])))
print(" %-14s %s / %s" % ("GBS / GA", d.get("gbs"), d.get("ga")))
print(" %-14s %s 초" % ("step 중앙값", f("step_median")))
print(" %-14s %s 초" % ("step 평균",   f("step_mean")))
print(" %-14s %s / %s / %s 초   (편차 %s%%)" % ("최소/p90/최대",
      f("step_min"), f("step_p90"), f("step_max"), f("spread_pct","%.1f")))
print(" %-14s %s" % ("TFLOPS/GPU", f("tflops_per_gpu","%.1f")))
print(" %-14s %s" % ("tokens/s",   f("tokens_per_s","%.0f")))
print(" %-14s %s" % ("samples/s",  f("samples_per_s","%.2f")))
if d.get("walltime_s"): print(" %-14s %.1f 초" % ("전체 소요", d["walltime_s"]))
print(" %-14s %s" % ("종료상태", d.get("status") or "N/A"))
if d.get("status")=="aborted":
    print("                (지정한 step 수에 도달해 정상 종료. 수렴 런이 아니면 정상입니다)")
for sc,v in d.get("evals") or []:
    print("   학습품질(log_ppl) @ %s 샘플 : %.4f   (목표 3.3 미만)" % (sc,v))
# step 편차가 크면 측정이 불안정하다는 뜻 — step 수를 늘려야 합니다
sp=d.get("spread_pct")
if sp is not None and sp > 8:
    print("")
    print(" !! step 편차 %.1f%% — 측정이 불안정합니다. step 수를 늘려 재측정하세요 (perf 500)" % sp)
if syn:
    print("")
    print(" !! 가짜(synthetic) 데이터 결과입니다. 성능 수치로 쓰면 안 됩니다.")
    print("    가짜 토크나이저는 어휘 32000개(실제 128256개)라 모델이 약 0.8B")
    print("    파라미터 작습니다. 실제 데이터는 이보다 느립니다.")
m=d.get("step_median")
print("==================================================")
# 아래 한 줄만 옮겨 적으면 결과를 재구성할 수 있습니다 (로그 반출 제약 대응)
print(" CODE: V6T.L8B.G4.%s.%s%s" % ((("%.4f"%m) if m else "FAIL"),
      (("%.1f"%d["tflops_per_gpu"]) if d.get("tflops_per_gpu") else "0"),
      (".SYNTH" if syn else "")))
print("==================================================")
PYEOF
  printf " %-14s %s\n" "GPU MiB" "$(mem | tr '\n' ' ')"
  say " 상세: $L"
}

# ============================== 학습 실행기 ==================================
# 인자:  $1=설정파일  $2=화면에 표시할 제목  $3...=설정을 읽은 뒤 덮어쓸 값들
#
# 세 번째 인자부터가 중요합니다. 설정파일을 읽은 "뒤에" 적용하기 때문에
# MAX_STEPS 만 바꾸고 학습률 계획은 건드리지 않습니다 (앞서 (5)번 문제).
launch(){
  local cfg="$1"; shift
  local tag="$1"; shift

  load_cfg "$cfg" || return 1
  for kv in "$@"; do export "${kv?}"; done      # 덮어쓸 값 적용

  export CONT DATADIR LOGDIR
  export NEXP="${NEXP:-1}"                      # 반복 실험 횟수
  export SEED="${SEED:-1234}"
  export MLPERF_CLUSTER_NAME="${MLPERF_CLUSTER_NAME:-XE9780}"
  export MLPERF_SCALE="${MLPERF_SCALE:-single_node_4gpu}"
  export CLEAR_CACHES="${CLEAR_CACHES:-0}"      # 1 은 sudo sysctl 권한 필요
  # 앞서 (3)번 문제: 컨테이너 OpenMPI 가 PMIx 를 요구하나 이 SLURM 에는 없음.
  # 학습 자체는 SLURM 랭크를 직접 쓰므로 무관하지만 로그가 지저분해집니다.
  export NCCL_TEST="${NCCL_TEST:-0}"
  export NCCL_LLM_TEST="${NCCL_LLM_TEST:-0}"
  [ "${NO_BIND:-0}" = "1" ] && export BINDCMD=""

  # 이번 실행의 로그 파일을 정확히 찾기 위해 타임스탬프를 직접 정합니다
  export DATESTAMP="${DATESTAMP:-$(date +'%y%m%d%H%M%S%N')}"

  # 앞서 (4)번 문제: sbatch 는 제출 쉘의 환경변수를 컨테이너까지 넘깁니다.
  # 호스트 경로가 컨테이너 안 torch/cuda 라이브러리를 가리면 학습이 깨집니다.
  if [ "${KEEP_HOST_ENV:-0}" != "1" ]; then
    for e in LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME CUDA_HOME \
             CPATH LIBRARY_PATH NCCL_ROOT MPI_HOME OPAL_PREFIX; do
      if [ -n "${!e:-}" ]; then say "   호스트 ${e} 제거: ${!e}"; unset "$e"; fi
    done
  fi
  # 앞서 (2)번 문제: cuDNN 위치를 직접 알려줍니다
  export CUDNN_PATH="${CUDNN_PATH:-$CUDNN_PATH_DEFAULT}"

  say ""
  say "=================================================="
  say " $tag"
  say "=================================================="
  show_cfg
  printf " %-22s %s\n" "CUDNN_PATH"  "$CUDNN_PATH"
  printf " %-22s %s\n" "덮어쓴 값"   "$*${NO_BIND:+ NO_BIND=1}"
  printf " %-22s %s\n" "로그 폴더"   "$LOGDIR"
  hr

  # 앞서 (1)번 문제: 컨테이너 이미지 형식 검증
  case "$CONT" in
    /*.sqsh|/*.squashfs)
      [ -s "$CONT" ] || { ng "컨테이너 파일이 없습니다: $CONT"
                          say "      bash $0 build   또는"
                          say "      enroot import -o $CONT dockerd://$DOCKER_TAG"
                          return 1; }
      ok "컨테이너 = $CONT ($(du -h "$CONT" 2>/dev/null | cut -f1))" ;;
    dockerd://*|docker://*)
      ok "컨테이너 = $CONT  (매 실행마다 변환하므로 .sqsh 권장)" ;;
    *)
      ng "컨테이너 경로 형식 오류: $CONT"
      say "      이름만 적으면 pyxis 가 Docker Hub 로 찾아가 401 오류가 납니다."
      say "      export CONT=$BASE/llama31_8b.sqsh   (bash $0 build 로 생성)"
      return 1 ;;
  esac

  hard_clean || return 1

  local GRES_OPT=()
  [ "${USE_GRES:-1}" = "1" ] && GRES_OPT=( --gres="gpu:${DGXNGPU}" )

  cd "$REPO" || return 1
  local JOB SOUT
  # sbatch 로 던집니다. 터미널을 닫아도 학습은 계속 돕니다.
  JOB=$(sbatch --parsable -N "${DGXNNODES}" "${GRES_OPT[@]}" --time="${WALLTIME}" \
        --output="${LOGDIR}/slurm-%j.out" --error="${LOGDIR}/slurm-%j.out" \
        ${SLURM_EXTRA:-} run.sub 2>>"$V")
  if [ -z "$JOB" ]; then
    ng "작업 제출 실패"; tail -20 "$V" | sed 's/^/   /'; return 1
  fi
  SOUT="${LOGDIR}/slurm-${JOB}.out"
  say " 작업번호 $JOB   출력: $SOUT"
  say " 15초마다 상태를 표시합니다."

  # --- 진행 감시 ------------------------------------------------------------
  local t=0 st
  while :; do
    st=$(squeue -h -j "$JOB" -o '%T' 2>/dev/null)
    [ -z "$st" ] && break                       # 큐에서 사라지면 종료된 것
    if [ "$st" = "PENDING" ]; then
      printf '   %s  [대기중: %s]\n' "$(date +%H:%M:%S)" \
             "$(squeue -h -j "$JOB" -o '%r' 2>/dev/null)"
    else
      printf '   %s  [%s]  GPU MiB [%s]\n' "$(date +%H:%M:%S)" "$st" \
             "$(mem | tr '\n' ' ')"
    fi
    sleep 15; t=$((t+1))
  done
  say " 작업 $JOB 종료 (약 $((t/4))분 경과)"
  sleep 5

  # --- 성공 판정 ------------------------------------------------------------
  # 앞서 (6)번 문제: SLURM 상태가 아니라 학습 로그 내용으로 판단합니다.
  # sacct -X 는 작업 전체 상태만 보여줍니다(내부 단계 실패는 무시).
  local MLLOG; MLLOG=$(find_mllog "$DATESTAMP")
  if [ -n "$MLLOG" ] && has_steps "$MLLOG"; then
    say ""
    sacct -X -n -j "$JOB" -o State,ExitCode,Elapsed 2>/dev/null \
      | sed 's/^/   SLURM: /' | head -1
    report "$MLLOG"
    unset DATESTAMP
    return 0
  fi

  # --- 실패했을 때: 원인을 찾을 단서를 모아 보여줍니다 ---------------------
  ng "학습 step 이 기록되지 않았습니다."
  say ""
  say "[SLURM 상태]"
  sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,Reason%30 2>/dev/null | sed 's/^/   /'
  say ""
  say "[출력 마지막 50줄]  $SOUT"
  if [ -f "$SOUT" ]; then
    tail -50 "$SOUT" | sed 's/^/   /'
    say ""
    say "[치명적 오류 후보]  (MPI_Init / pmix / cpuset.mems 는 무해합니다)"
    grep -inE 'Traceback|RuntimeError|ImportError|ModuleNotFound|CUDA (error|out of memory)|OutOfMemory|shared object not found|401 Unauthorized|couldn.t start container' \
      "$SOUT" 2>/dev/null | head -12 | sed 's/^/   /'
  else
    ng "출력 파일이 없습니다: $SOUT"
  fi
  say ""
  say " 점검:  bash $0 doctor    또는    bash $0 inspect"
  unset DATESTAMP
  return 1
}

# =============================================================================
#                              명령어 처리
# =============================================================================
case "$MODE" in

# -------------------------------------------------------------------- clean --
clean) hard_clean; squeue -u "$(whoami)" 2>/dev/null ;;

# ------------------------------------------------------------------- result --
# 학습을 다시 하지 않고 최근 결과만 다시 표시합니다.
result)
  load_cfg "${A2:-$CFG_BASE}" >/dev/null 2>&1
  report ;;

# -------------------------------------------------------------------- check --
# 실행 전 필요한 것이 다 있는지 점검합니다.
check)
  say "MLPerf Training v6.0  Llama 3.1 8B  사전점검"
  say ""
  say "[소스코드]"
  [ -d "$REPO" ] && ok "위치 $REPO" || ng "소스코드가 없습니다: $REPO"
  for f in run.sub run_and_time.sh config_mounts.sh pretrain.py conf/custom.yaml; do
    [ -f "$REPO/$f" ] && ok "$f" || ng "누락: $f"
  done
  say ""
  say "[설정파일]"
  for c in "$CFG_BASE" "$CFG_ATTN"; do
    if [ -f "$REPO/$c" ]; then
      bash -n "$REPO/$c" 2>/dev/null && ok "$c" || ng "$c 문법 오류"
    else
      ng "$c 없음 — 소스코드 폴더에 복사하세요"
    fi
  done
  # 설정파일이 참조하는 공통 파일들
  for c in config_common.sh config_common_fp4.sh config_common_cg.sh \
           config_common_8b.sh config_common_fp8attn.sh; do
    [ -f "$REPO/$c" ] || ng "의존 파일 없음: $c"
  done
  say ""
  say "[학습 데이터]  $DATADIR/8b"
  if [ -d "$DATADIR/8b" ]; then
    for f in c4-train.en_6_text_document.bin c4-train.en_6_text_document.idx \
             c4-validation-91205-samples.en_text_document.bin \
             c4-validation-91205-samples.en_text_document.idx \
             tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
      [ -s "$DATADIR/8b/$f" ] && ok "$f ($(du -h "$DATADIR/8b/$f" | cut -f1))" \
                              || ng "누락: $f"
    done
  else
    ng "없습니다 — bash $0 prep"
  fi
  say "  디스크 여유 $(df -BG --output=avail "$DATADIR" 2>/dev/null | tail -1 | tr -d ' ')  (데이터 약 90GB 필요)"
  say ""
  say "[컨테이너]"
  case "$CONT" in
    /*.sqsh|/*.squashfs)
      [ -s "$CONT" ] && ok "$CONT ($(du -h "$CONT" | cut -f1))" \
                     || ng "$CONT 없습니다 — bash $0 build" ;;
    *) ng "경로 형식 오류: $CONT  (.sqsh 절대경로가 필요합니다)" ;;
  esac
  say ""
  say "[스케줄러 / GPU]"
  command -v sbatch >/dev/null 2>&1 && ok "sbatch" || ng "sbatch 없음"
  srun --help 2>/dev/null | grep -q container-image && ok "pyxis" || ng "pyxis 없음"
  command -v enroot >/dev/null 2>&1 && ok "enroot" || ng "enroot 없음"
  say "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) x $(nvidia-smi -L 2>/dev/null | wc -l)장"
  say "  GPU MiB [$(mem | tr '\n' ' ')]   대기작업 $(squeue -h -u "$(whoami)" 2>/dev/null | wc -l)개"
  say ""
  if load_cfg "$CFG_BASE" 2>/dev/null; then say "[해석된 설정]"; show_cfg; fi
  say ""
  say " 다음:  bash $0 dryrun  ->  bash $0 smoke  ->  bash $0 perf"
  ;;

# ------------------------------------------------------------------- doctor --
# 환경을 더 깊이 진단하고, 최근 실패의 원인 단서를 모아 보여줍니다.
doctor)
  say "실행환경 진단"
  say ""
  say "[스케줄러]"
  command -v sbatch >/dev/null 2>&1 && ok "sbatch" || ng "sbatch 없음"
  command -v sacct  >/dev/null 2>&1 && ok "sacct"  || ng "sacct 없음 (종료상태 확인 제한)"
  sinfo -h -o '   노드=%n 상태=%t CPU=%c GPU=%G' 2>/dev/null | sed 's/^/  /'
  say ""
  say "[컨테이너 도구]"
  srun --help 2>/dev/null | grep -q container-image && ok "pyxis" || ng "pyxis 없음"
  command -v enroot >/dev/null 2>&1 && ok "enroot $(enroot version 2>/dev/null)" || ng "enroot 없음"
  say ""
  say "[컨테이너 이미지]  $CONT"
  case "$CONT" in
    /*.sqsh|/*.squashfs)
      [ -s "$CONT" ] && ok "존재 ($(du -h "$CONT" | cut -f1))" || ng "파일 없음" ;;
    dockerd://*|docker://*) ok "URI 형식 (매번 변환됨)" ;;
    *) ng "pyxis 는 로컬 docker 이미지를 못 읽습니다 — .sqsh 절대경로 필요" ;;
  esac
  say ""
  say "[호스트 환경변수 — 컨테이너로 새어 들어갑니다]"
  for e in LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH CUDA_HOME CUDNN_PATH; do
    printf '   %-18s [%s]\n' "$e" "${!e:-없음}"
  done
  say "   (실행 시 자동으로 제거합니다. KEEP_HOST_ENV=1 로 유지 가능)"
  say ""
  say "[GPU]"
  nvidia-smi -L 2>/dev/null | sed 's/^/   /'
  say "   요청 예정 ${NGPU}장 / 보이는 GPU $(nvidia-smi -L 2>/dev/null | wc -l)장"
  say ""
  say "[최근 작업]"
  sacct -X -n -u "$(whoami)" -S "$(date -d '2 days ago' +%Y-%m-%d)" \
        --format=JobID%10,State%12,ExitCode%8,Elapsed%10,Reason%28 2>/dev/null \
    | tail -5 | sed 's/^/   /'
  say ""
  say "[최근 출력에서 발견된 오류]"
  for f in $(ls -1t "$LOGDIR"/slurm-*.out "$REPO"/slurm-*.out 2>/dev/null | head -1); do
    say "   --- $f"
    grep -inE 'Traceback|RuntimeError|ImportError|shared object not found|401 Unauthorized|couldn.t start container|out of memory' \
      "$f" 2>/dev/null | head -10 | sed 's/^/     /' || say "     (치명적 오류 없음)"
  done
  ;;

# ------------------------------------------------------------------ inspect --
# 컨테이너 안에 들어가 cuDNN / TransformerEngine 상태를 직접 확인합니다.
# 앞서 (2)번 문제를 진단하는 전용 모드입니다.
inspect)
  case "$CONT" in
    /*.sqsh|/*.squashfs) [ -s "$CONT" ] || { ng "컨테이너 없음: $CONT"; exit 1; } ;;
    dockerd://*|docker://*) ;;
    *) ng "컨테이너 경로 형식 오류: $CONT"; exit 1 ;;
  esac
  say "컨테이너 내부 점검   CONT=$CONT"
  say ""
  # 호스트 환경변수를 제거한 상태로 들어가야 정확히 진단됩니다
  env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u CUDA_HOME -u CUDNN_PATH \
  srun --ntasks=1 -N1 --time=10 --container-image="$CONT" \
       --no-container-mount-home --container-remap-root --container-writable \
       bash -c '
    echo "  LD_LIBRARY_PATH = [$LD_LIBRARY_PATH]"
    echo "  CUDNN_VERSION   = [${CUDNN_VERSION:-없음}]"
    echo "  --- libcudnn 파일 (버전 없는 libcudnn.so 가 있는지 확인)"
    ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so* 2>&1 | head -5
    echo "  --- 설치된 cudnn 패키지"
    dpkg -l 2>/dev/null | grep -i cudnn || echo "    (없음 — 이미지 재빌드 필요)"
    echo "  --- CUDNN_PATH 없이 import 시도"
    python -c "import transformer_engine.pytorch; print(\"    OK\")" 2>&1 | tail -1
    echo "  --- CUDNN_PATH 지정해서 import 시도"
    CUDNN_PATH=/usr/lib/x86_64-linux-gnu \
      python -c "import transformer_engine.pytorch; print(\"    OK\")" 2>&1 | tail -1
    python -c "import torch; print(\"  torch\", torch.__version__, \"cuda\", torch.version.cuda)"
  '
  say ""
  say " 두 번째 import 만 OK 면 이 스크립트가 이미 자동 처리합니다."
  say " 둘 다 실패하면 Dockerfile 에 아래를 넣고 재빌드하세요:"
  say "   RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9 \\"
  say "              /usr/lib/x86_64-linux-gnu/libcudnn.so"
  ;;

# -------------------------------------------------------------------- build --
# docker 로 이미지를 만들고, pyxis 가 읽을 수 있는 .sqsh 로 변환합니다.
build)
  say "컨테이너 이미지 만들기  |  상세로그 $V"
  cd "$REPO" || exit 1
  say "  베이스 이미지: $(grep -m1 '^ARG FROM_IMAGE_NAME' Dockerfile | cut -d= -f2)"
  say "  NGC 로그인이 없으면 여기서 401 오류가 납니다:"
  say "    echo \$NGC_API_KEY | docker login nvcr.io -u '\$oauthtoken' --password-stdin"
  say ""
  docker build -t "$DOCKER_TAG" \
    ${FROM_IMAGE_NAME:+--build-arg FROM_IMAGE_NAME=$FROM_IMAGE_NAME} \
    --build-arg GIT_COMMIT_ID="$(git rev-parse --short HEAD 2>/dev/null || echo na)" . \
    2>&1 | tee -a "$V" | tail -20
  docker image inspect "$DOCKER_TAG" >/dev/null 2>&1 \
    || { ng "빌드 실패 — $V 확인"; exit 1; }
  ok "빌드 완료: $DOCKER_TAG"
  say ""
  say ".sqsh 변환 (pyxis 는 로컬 docker 이미지를 못 읽습니다). 10~20분."
  # 임시 폴더 기본값 /tmp 는 메모리 디스크라 큰 이미지에서 터집니다
  export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-$BASE/enroot-tmp}"
  mkdir -p "$ENROOT_TEMP_PATH"
  say "  임시폴더=$ENROOT_TEMP_PATH (여유 $(df -BG --output=avail "$ENROOT_TEMP_PATH" 2>/dev/null | tail -1 | tr -d ' '))"
  rm -f "$CONT"
  enroot import -o "$CONT" "dockerd://$DOCKER_TAG" 2>&1 | tee -a "$V" | tail -8
  if [ -s "$CONT" ]; then
    ok ".sqsh 생성 완료: $CONT ($(du -h "$CONT" | cut -f1))"
    say " ~/.bashrc 에 추가하세요:  export CONT=$CONT"
  else
    ng ".sqsh 변환 실패 — $V 확인 (디스크 여유/도커 권한)"
  fi
  ;;

# --------------------------------------------------------------------- prep --
prep)
  say "학습 데이터 내려받기 (약 85GB)  |  대상 $DATADIR  상세로그 $V"
  mkdir -p "$DATADIR"
  cd "$REPO" || exit 1
  DATADIR="$DATADIR" bash data_scripts/download_8b.sh 2>&1 | tee -a "$V" | tail -20
  say ""; ls -lh "$DATADIR/8b" 2>/dev/null | sed 's/^/   /'
  say ""; say " 검증: bash $0 check"
  ;;

# --------------------------------------------------------------------- nccl --
# GPU 간 통신 속도만 측정합니다. PMIx 가 동작하는 환경에서만 의미가 있습니다.
nccl)
  launch "$CFG_BASE" "GPU 간 통신속도 측정" \
    RUN_ONLY_NCCL=1 NCCL_TEST=1 NCCL_TEST_WALLTIME=10 \
    USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 CHECK_COMPLIANCE=0 \
    WALLTIME_RUNANDTIME=15 WALLTIME=20
  ;;

# ------------------------------------------------------------------- dryrun --
# 실제 학습 없이, 최종 적용될 설정만 출력합니다. 설정 검증용.
dryrun)
  launch "$CFG_BASE" "DRYRUN — 최종 설정만 출력" \
    PRINT_CONFIG_ONLY=True USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 \
    MAX_STEPS=1 CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=20 WALLTIME=25
  say " 출력에서 model.global_batch_size / trainer.val_check_interval 을 확인하세요"
  ;;

# -------------------------------------------------------------------- smoke --
# 가짜 데이터로 파이프라인이 끝까지 도는지만 확인합니다.
# 성능 수치는 의미가 없습니다 (가짜 토크나이저라 모델이 더 작음).
smoke)
  launch "$CFG_BASE" "SMOKE — 가짜 데이터 ${A2:-50} step (동작 확인용, 성능 무의미)" \
    USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 CHECK_COMPLIANCE=0 \
    MAX_STEPS="${A2:-50}" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    WALLTIME_RUNANDTIME=40 WALLTIME=45
  ;;

# --------------------------------------------------------------------- perf --
# 실제 데이터로 속도를 측정합니다. 이게 본론입니다.
# 평가 주기(768 step)가 MAX_STEPS 보다 크면 평가가 안 돌아서
# 순수한 학습 step 시간만 측정됩니다.
perf)
  launch "$CFG_BASE" "PERF — 실제 데이터 ${A2:-200} step" \
    MAX_STEPS="${A2:-200}" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=60 WALLTIME=70
  ;;

# ------------------------------------------------------------------ fp8attn --
# 어텐션 연산을 FP8 로 바꿔서 perf 결과와 비교합니다 (A/B 테스트).
fp8attn)
  launch "$CFG_ATTN" "PERF (FP8 어텐션) — ${A2:-200} step   <- perf 결과와 비교" \
    MAX_STEPS="${A2:-200}" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=60 WALLTIME=70
  ;;

# -------------------------------------------------------------------- sweep --
# MICRO_BATCH_SIZE 를 1/2/4 로 바꿔가며 자동 비교합니다.
# 이 값은 전체 배치크기(GBS)를 바꾸지 않는 유일한 성능 조절 항목입니다.
# MINIBS=4 이므로 1/2/4 만 GA 가 정수가 됩니다.
# MBS=4 는 GA=1 이 되어 Dell 의 1x8x2 제출 조건과 같아집니다.
sweep)
  S="${A2:-150}"
  RES=()
  for MBS in 1 2 4; do
    say ""; say "########## MICRO_BATCH_SIZE=$MBS  (GA=$((4/MBS))) ##########"
    if launch "$CFG_BASE" "SWEEP MBS=${MBS}" \
        MICRO_BATCH_SIZE="$MBS" MAX_STEPS="$S" LOG_EVERY_N_STEPS=1 \
        LOG_METRICS=DELTA CHECK_COMPLIANCE=0 \
        WALLTIME_RUNANDTIME=50 WALLTIME=60; then
      L=$(find_mllog)
      T=$(python3 -c "$PARSE_PY" "$L" "$NGPU" "$TFLOP_PER_SAMPLE" "$SEQLEN" 2>/dev/null \
          | python3 -c "import json,sys;d=json.load(sys.stdin);print('%.4f %.1f'%(d.get('step_median') or 0,d.get('tflops_per_gpu') or 0))")
      RES+=("MBS=$MBS GA=$((4/MBS))  step=$(echo $T|cut -d' ' -f1)초  TFLOPS/GPU=$(echo $T|cut -d' ' -f2)")
    else
      RES+=("MBS=$MBS  실패")
    fi
  done
  say ""
  say "=================================================="
  say " MICRO_BATCH_SIZE 비교 결과 (GBS=16 고정)"
  say "=================================================="
  for r in "${RES[@]}"; do say "   $r"; done
  say "=================================================="
  ;;

# ----------------------------------------------------------------- converge --
# 실제 목표 품질(log_ppl 3.3)까지 학습합니다.
# GPU 4장(GBS=16)에서는 수십~수백 시간 규모입니다.
converge)
  say "수렴 학습: 목표 log_ppl 3.3"
  say "GPU 4장(GBS=16)에서는 수십~수백 시간이 걸립니다."
  say "중단되어도 이어서 재개되도록 재시작 옵션을 켭니다."
  launch "$CFG_BASE" "CONVERGE — log_ppl 3.3 목표" \
    ENABLE_RERUNS=1 SHARE_RERUNS=1 STORE_CKPTS_IN_LOGDIR=1 \
    WALLTIME_EXIT_MINUTES=10
  ;;

# 인자 없이 실행하면 맨 위 사용법을 보여줍니다
*) sed -n '3,40p' "$0" ;;
esac
