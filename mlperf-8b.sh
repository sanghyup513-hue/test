#!/usr/bin/env bash
# =============================================================================
# MLPerf Training v6.0 - Llama 3.1 8B (NeMo)
# Dell XE9780 / B300  -  1 node x 4 GPU  -  NVFP4  (GBS=16, GA=2)
#
#   bash mlperf-8b.sh clean                잔여 잡/컨테이너/GPU 정리
#   bash mlperf-8b.sh check                자산·config·환경 점검
#   bash mlperf-8b.sh doctor               실행환경 진단 (pyxis/enroot/CONT/잡 실패원인)
#   bash mlperf-8b.sh inspect              컨테이너 내부 점검 (cuDNN / TransformerEngine)
#   bash mlperf-8b.sh build                컨테이너 빌드
#   bash mlperf-8b.sh prep                 데이터셋 + 토크나이저 다운로드 (~85GB)
#
#   bash mlperf-8b.sh dryrun               resolved config만 출력 (GBS 확인, GPU 거의 안 씀)
#   bash mlperf-8b.sh smoke                synthetic data 50 step 기능 확인 (데이터셋 불필요)
#   bash mlperf-8b.sh perf                 성능 측정 (기본 200 step, eval 없음)
#   bash mlperf-8b.sh perf 500             step 수 지정
#   bash mlperf-8b.sh fp8attn              FP8 attention A/B 런
#   bash mlperf-8b.sh sweep                MICRO_BATCH_SIZE 1/2/4 스윕
#   bash mlperf-8b.sh converge             수렴 런 (log_ppl 3.3 목표, 매우 김)
#   bash mlperf-8b.sh result               최근 결과 재파싱
#
# 성능 지표는 mllog의 tracked_stats.train_step_time 에서 추출합니다.
#   TFLOPS/GPU = MODEL_TFLOP_PER_SAMPLE(421.59) * GBS / step_time / NGPU
#   tokens/s   = GBS * 8192 / step_time
#
# 주의: MAX_STEPS 는 config 를 source 한 "뒤에" 덮어씁니다.
#       config 안에서 바꾸면 OPT_LR_DECAY_STEPS 까지 따라 바뀌어 LR 궤적이 왜곡됩니다.
# =============================================================================
set -u

BASE="${BASE:-/data/lsh}"
REPO="${REPO:-$BASE/training_results_v6.0/Dell/benchmarks/llama31_8b/implementations/nemo}"
DATADIR="${DATADIR:-$BASE/mlperf_training_data}"
LOGDIR="${LOGDIR:-$BASE/mlperf-8b-logs}"
CONT="${CONT:-$BASE/llama31_8b.sqsh}"          # pyxis 용 sqsh. bare 이미지명은 동작하지 않음
DOCKER_TAG="${DOCKER_TAG:-mlperf-nvidia:llama31_8b-pyt}"  # docker build 결과 태그
CFG_BASE="config_XE9780_B300_1x4x4xtp1pp1cp1_8b_fp4.sh"
CFG_ATTN="config_XE9780_B300_1x4x4xtp1pp1cp1_8b_fp4_fp8attn.sh"
NGPU=4
SEQLEN=8192

MODE="${1:-}"; A2="${2:-}"; A3="${3:-}"
V="$BASE/8b-$(date +%m%d-%H%M).log"
mkdir -p "$LOGDIR" 2>/dev/null

say(){ printf '%s\n' "$*"; }
ok(){  printf '   OK   %s\n' "$*"; }
ng(){  printf '   --   %s\n' "$*"; }
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; }

# ---------------------------------------------------------------------------
hard_clean(){
  scancel -u "$(whoami)" >>"$V" 2>&1; sleep 3
  local i pids p
  for i in 1 2 3 4 5 6; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
  # pyxis 가 남긴 컨테이너 제거
  if command -v enroot >/dev/null 2>&1; then
    [ -n "$(enroot list 2>/dev/null)" ] && enroot remove -f $(enroot list) >>"$V" 2>&1
  fi
  for i in 1 2 3 4 5 6; do
    squeue -h -u "$(whoami)" 2>/dev/null | grep -q . || break
    sleep 5
  done
  local busy; busy=$(mem | awk '$1>1000{n++}END{print n+0}')
  if [ "${busy:-0}" -gt 0 ]; then
    ng "GPU 메모리 미해제: [$(mem | tr '\n' ' ')]"
    ng "확인: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    return 1
  fi
  ok "GPU 전부 비움 [$(mem | tr '\n' ' ')]   대기잡 $(squeue -h -u "$(whoami)" 2>/dev/null | wc -l)개"
  return 0
}

# ---------------------------------------------------------------------------
# config 를 source 하고 파생값을 계산해 전역에 export
load_cfg(){   # $1 = config 파일명
  local c="$REPO/$1"
  [ -f "$c" ] || { ng "config 없음: $c"; return 1; }
  set -a; source "$c"; set +a
  MP=$(( TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL ))
  WS=$(( DGXNNODES * DGXNGPU ))
  DP=$(( WS / MP ))
  GBS=$(( MINIBS * DP ))
  GA=$(( MINIBS / MICRO_BATCH_SIZE ))
  export MP WS DP GBS GA
  return 0
}

show_cfg(){
  printf " %-22s %s\n" "config"      "$CFG"
  printf " %-22s %s\n" "DGXSYSTEM"   "$DGXSYSTEM"
  printf " %-22s %s\n" "topology"    "${DGXNNODES}node x ${DGXNGPU}gpu = WS ${WS}"
  printf " %-22s %s\n" "parallelism" "TP${TENSOR_MODEL_PARALLEL} PP${PIPELINE_MODEL_PARALLEL} CP${CONTEXT_PARALLEL} -> DP${DP}"
  printf " %-22s %s\n" "batch"       "MINIBS ${MINIBS} x DP ${DP} = GBS ${GBS}   (MBS ${MICRO_BATCH_SIZE}, GA ${GA})"
  printf " %-22s %s\n" "precision"   "FP4=${FP4} recipe=${FP4_RECIPE} FP8_DPA=${FP8_DPA:-False} CUDA_GRAPH=${MCORE_CUDA_GRAPH:-0}"
  printf " %-22s %s\n" "lr/warmup"   "LR ${LR}  WARMUP ${WARMUP_STEPS}  decay_steps ${OPT_LR_DECAY_STEPS}"
  printf " %-22s %s\n" "eval"        "VAL_CHECK_INTERVAL ${VAL_CHECK_INTERVAL} (auto=$(( (12288 + GBS - 1) / GBS )))  VAL_SAMPLES ${VAL_SAMPLES}"
  printf " %-22s %s\n" "max_steps"   "${MAX_STEPS}"
  # GBS 정합성 자체 검사
  local auto=$(( (12288 + GBS - 1) / GBS ))
  [ "$VAL_CHECK_INTERVAL" -ne "$auto" ] && \
    ng "VAL_CHECK_INTERVAL(${VAL_CHECK_INTERVAL}) != ceil(12288/GBS)=${auto}  -> eval 주기 규칙과 불일치"
  [ $(( MINIBS % MICRO_BATCH_SIZE )) -ne 0 ] && \
    ng "MINIBS 가 MICRO_BATCH_SIZE 로 나누어떨어지지 않습니다 (GA 정수 아님)"
  return 0
}

# ---------------------------------------------------------------------------
PARSE_PY='
import json,sys,statistics as st
path,ngpu,tflop = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
steps=[]; evals=[]; gbs=None; ga=None; t0=t1=None; status=None; seq=8192
for line in open(path, errors="ignore"):
    i=line.find(":::MLLOG")
    if i<0: continue
    try: d=json.loads(line[i+8:])
    except Exception: continue
    k=d.get("key"); v=d.get("value"); m=d.get("metadata") or {}
    if k=="tracked_stats" and isinstance(v,dict) and "train_step_time" in v:
        t=v["train_step_time"]
        if t and t>0: steps.append(t)
    elif k=="eval_accuracy": evals.append((m.get("samples_count"), v))
    elif k=="global_batch_size": gbs=v
    elif k=="gradient_accumulation_steps": ga=v
    elif k=="run_start": t0=d.get("time_ms")
    elif k=="run_stop":  t1=d.get("time_ms"); status=m.get("status")
warm=min(10,max(0,len(steps)//5))
body=steps[warm:] or steps
out={"n_step":len(steps),"n_used":len(body),"gbs":gbs,"ga":ga,"status":status,
     "evals":evals[-5:],"walltime_s":(t1-t0)/1000.0 if (t0 and t1) else None}
if body:
    med=st.median(body)
    out.update({
      "step_median":med,"step_mean":sum(body)/len(body),
      "step_min":min(body),"step_max":max(body),
      "step_p90":sorted(body)[int(len(body)*0.9)-1] if len(body)>=10 else max(body),
      "tflops_per_gpu": (tflop*(gbs or 0)/med/ngpu) if gbs else None,
      "tokens_per_s":   ((gbs or 0)*seq/med) if gbs else None,
      "samples_per_s":  (gbs or 0)/med if gbs else None,
    })
print(json.dumps(out))
'

report(){   # $1 = 로그파일(옵션)
  local L="${1:-}" J
  if [ -z "$L" ]; then
    L=$(ls -1t "$LOGDIR"/*_0*.log 2>/dev/null | grep -v mountcheck | grep -v hang_monitor | grep -v 'slurm-' | head -1)
  fi
  if [ -z "$L" ] || [ ! -f "$L" ]; then ng "파싱할 로그가 없습니다 ($LOGDIR)"; return 1; fi

  J=$(python3 -c "$PARSE_PY" "$L" "$NGPU" "${MODEL_TFLOP_PER_SAMPLE:-421.59}" 2>/dev/null)
  say ""
  say "=================================================="
  say " llama31_8b  GPU ${NGPU}장 (TP1 PP1 CP1 x DP${DP:-4})  GBS ${GBS:-?}"
  say "=================================================="
  printf " %-12s %s\n" "로그" "$(basename "$L")"
  if [ -z "$J" ]; then
    ng "mllog 파싱 실패 - 아래 원인 확인"
    grep -m5 -iE "error|traceback|out of memory|CUDA|assert" "$L" | sed 's/^/   /'
    return 1
  fi
  python3 - "$J" <<'PY'
import json,sys
d=json.loads(sys.argv[1])
f=lambda k,fmt="%.4f",dflt="N/A": (fmt%d[k]) if d.get(k) is not None else dflt
print(" %-12s %s"   % ("측정step", f"{d['n_used']} / {d['n_step']} (앞 warmup 제외)"))
print(" %-12s %s"   % ("GBS/GA",   f"{d.get('gbs')} / {d.get('ga')}"))
print(" %-12s %s s" % ("step 중앙값", f("step_median")))
print(" %-12s %s s" % ("step 평균",   f("step_mean")))
print(" %-12s %s / %s s" % ("step min/p90", f("step_min"), f("step_p90")))
print(" %-12s %s" % ("TFLOPS/GPU", f("tflops_per_gpu","%.1f")))
print(" %-12s %s" % ("tokens/s",   f("tokens_per_s","%.0f")))
print(" %-12s %s" % ("samples/s",  f("samples_per_s","%.2f")))
if d.get("walltime_s"): print(" %-12s %.1f s" % ("run_start->stop", d["walltime_s"]))
print(" %-12s %s" % ("run_stop", d.get("status") or "N/A"))
for sc,v in d.get("evals") or []:
    print("   eval_accuracy(log_ppl) @ %s samples : %.4f  (target < 3.3)" % (sc,v))
m=d.get("step_median")
code = "V6T.L8B.G4.%s.%s" % (("%.4f"%m) if m else "FAIL", ("%.1f"%d["tflops_per_gpu"]) if d.get("tflops_per_gpu") else "0")
print("==================================================")
print(" CODE: "+code)
print("==================================================")
PY
  printf " %-12s %s\n" "GPU MiB" "$(mem | tr '\n' ' ')"
  say " 상세: $L"
}

# ---------------------------------------------------------------------------
# 공통 런처. config source -> perf 오버라이드 -> sbatch run.sub
launch(){   # $1=config  $2=태그  $3...=추가 export (KEY=VAL)
  local cfg="$1"; shift
  local tag="$1"; shift

  load_cfg "$cfg" || return 1
  CFG="$cfg"

  # ---- config source 이후에만 덮어쓸 것 (OPT_LR_DECAY_STEPS 보존) ----
  for kv in "$@"; do export "${kv?}"; done

  export CONT DATADIR LOGDIR
  export NEXP="${NEXP:-1}"
  export MLPERF_CLUSTER_NAME="${MLPERF_CLUSTER_NAME:-XE9780}"
  export MLPERF_SCALE="${MLPERF_SCALE:-single_node_4gpu}"
  # sudo 없는 환경 기본값. 제출용이면 1 로.
  export CLEAR_CACHES="${CLEAR_CACHES:-0}"
  export NCCL_TEST="${NCCL_TEST:-1}"
  export NCCL_TEST_WALLTIME="${NCCL_TEST_WALLTIME:-5}"
  export SEED="${SEED:-1234}"
  # 이 클러스터는 pmix 미지원(OMPI not built with SLURM PMI). 405B 런과 동일하게 pmi2.
  export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmi2}"
  [ "${NO_BIND:-0}" = "1" ] && export BINDCMD=""

  # sbatch 는 --export=ALL 이 기본이라 제출 쉘의 환경이 컨테이너까지 들어갑니다.
  # 호스트 경로가 이미지의 라이브러리 탐색을 가려 "cudnn shared object not found" 를
  # 유발하므로 명시적으로 걷어냅니다. (KEEP_HOST_ENV=1 로 무력화 가능)
  if [ "${KEEP_HOST_ENV:-0}" != "1" ]; then
    for e in LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME CUDA_HOME CUDNN_PATH \
             CPATH LIBRARY_PATH NCCL_ROOT MPI_HOME OPAL_PREFIX; do
      if [ -n "${!e:-}" ]; then
        say "   호스트 ${e} 제거: ${!e}"
        unset "$e"
      fi
    done
  fi
  # 이미지 안에서 cuDNN 탐색이 실패할 때의 우회 (CUDNN_LIBDIR=... 로 지정)
  [ -n "${CUDNN_LIBDIR:-}" ] && export CUDNN_PATH="$CUDNN_LIBDIR" \
                             && export LD_LIBRARY_PATH="$CUDNN_LIBDIR"

  say ""
  say "=================================================="
  say " ${tag}"
  say "=================================================="
  show_cfg
  printf " %-22s %s\n" "overrides" "$* ${NO_BIND:+NO_BIND=1}"
  printf " %-22s %s\n" "LOGDIR"    "$LOGDIR"
  say "--------------------------------------------------"

  # ---- CONT 사전 검증: pyxis 는 로컬 docker 데몬을 보지 않습니다 ----
  case "$CONT" in
    /*.sqsh|/*.squashfs)
      if [ ! -s "$CONT" ]; then
        ng "CONT 파일이 없습니다: $CONT"
        say "      enroot import -o $CONT dockerd://mlperf-nvidia:llama31_8b-pyt"
        return 1
      fi
      ok "CONT = $CONT ($(du -h "$CONT" 2>/dev/null | cut -f1))" ;;
    dockerd://*|docker://*)
      ok "CONT = $CONT" ;;
    *)
      ng "CONT 형식이 잘못되었습니다: $CONT"
      say ""
      say "   접두사 없는 이미지명은 pyxis 가 Docker Hub 로 보내서 401 이 납니다."
      say "   (registry-1.docker.io/v2/library/... 401 Unauthorized)"
      say ""
      say "   해결:"
      say "     export ENROOT_TEMP_PATH=${BASE}/enroot-tmp && mkdir -p \$ENROOT_TEMP_PATH"
      say "     enroot import -o ${BASE}/llama31_8b.sqsh dockerd://${CONT}"
      say "     export CONT=${BASE}/llama31_8b.sqsh"
      say ""
      say "   빠른 확인용으로 dockerd://${CONT} 를 쓸 수도 있으나,"
      say "   매 잡마다 import 를 다시 하므로 sqsh 를 권장합니다."
      return 1 ;;
  esac

  hard_clean || return 1

  local GRES_OPT=()
  [ "${USE_GRES:-1}" = "1" ] && GRES_OPT=( --gres="gpu:${DGXNGPU}" )

  cd "$REPO" || return 1
  local JOB SOUT
  # run.sub 에는 #SBATCH --output 이 없어 기본값이 제출 디렉터리의 slurm-%j.out 입니다.
  # 실패 원인을 보려면 반드시 잡아둬야 합니다.
  JOB=$(sbatch --parsable -N "${DGXNNODES}" "${GRES_OPT[@]}" \
        --time="${WALLTIME}" \
        --output="${LOGDIR}/slurm-%j.out" --error="${LOGDIR}/slurm-%j.out" \
        ${SLURM_EXTRA:-} run.sub 2>>"$V")
  if [ -z "$JOB" ]; then
    ng "sbatch 제출 실패"
    tail -20 "$V" | sed 's/^/   /'
    return 1
  fi
  SOUT="${LOGDIR}/slurm-${JOB}.out"
  say " JobID $JOB  제출됨   stdout: $SOUT"
  say " 15초마다 상태를 표시합니다."

  local t=0 st reason
  while :; do
    st=$(squeue -h -j "$JOB" -o '%T' 2>/dev/null)
    [ -z "$st" ] && break
    if [ "$st" = "PENDING" ]; then
      reason=$(squeue -h -j "$JOB" -o '%r' 2>/dev/null)
      printf '   %s  [%s: %s]\n' "$(date +%H:%M:%S)" "$st" "$reason"
    else
      printf '   %s  [%s]  GPU MiB [%s]\n' "$(date +%H:%M:%S)" "$st" "$(mem | tr '\n' ' ')"
    fi
    sleep 15; t=$((t+1))
  done
  say " JobID $JOB 큐에서 내려감 (경과 약 $((t/4))분)"
  sleep 5

  # ---- 종료 상태 ----
  say ""
  say "[slurm 종료 상태]"
  sacct -j "$JOB" --format=JobID%16,State%14,ExitCode%8,Elapsed%10,Reason%30 2>/dev/null \
    | sed 's/^/   /' || ng "sacct 사용 불가"

  local FAILED=0
  sacct -n -j "${JOB}" -o State -P 2>/dev/null | grep -qiE 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_ME' && FAILED=1

  # ---- mllog 가 생겼는지 ----
  local MLLOG
  MLLOG=$(ls -1t "$LOGDIR"/*_0*.log 2>/dev/null | grep -v mountcheck | grep -v hang_monitor | grep -v '^.*slurm-' | head -1)

  if [ -z "$MLLOG" ] || [ "$FAILED" = "1" ]; then
    ng "run.sub 이 정상 완료되지 않았습니다."
    say ""
    say "[slurm stdout 마지막 60줄]  $SOUT"
    if [ -f "$SOUT" ]; then
      tail -60 "$SOUT" | sed 's/^/   /'
      say ""
      say "[에러 후보 라인]"
      grep -inE 'error|not found|no such file|permission denied|cannot|failed|invalid|refused|command not found|out of memory' \
        "$SOUT" 2>/dev/null | head -15 | sed 's/^/   /'
    else
      ng "stdout 파일이 없습니다: $SOUT  (sbatch 가 노드에 도달하지 못했을 수 있음)"
    fi
    say ""
    say " 점검:  bash $0 doctor"
    return 1
  fi

  report "$MLLOG"
}

# =============================================================================
case "$MODE" in

clean) hard_clean; squeue -u "$(whoami)" 2>/dev/null ;;

result)
  if [ -n "$A2" ]; then load_cfg "$A2" >/dev/null 2>&1; else load_cfg "$CFG_BASE" >/dev/null 2>&1; fi
  report ;;

check)
  say "MLPerf Training v6.0  Llama 3.1 8B  사전점검"
  say ""
  say "[경로]"
  [ -d "$REPO" ] && ok "REPO   $REPO" || ng "REPO 없음  $REPO"
  [ -f "$REPO/run.sub" ] && ok "run.sub" || ng "run.sub 없음"
  [ -f "$REPO/run_and_time.sh" ] && ok "run_and_time.sh" || ng "run_and_time.sh 없음"
  say ""
  say "[config]"
  for c in "$CFG_BASE" "$CFG_ATTN"; do
    if [ -f "$REPO/$c" ]; then
      bash -n "$REPO/$c" && ok "$c (문법 OK)" || ng "$c 문법 오류"
    else ng "$c 없음 - REPO 에 복사하세요"; fi
  done
  for c in config_common.sh config_common_fp4.sh config_common_cg.sh config_common_8b.sh config_common_fp8attn.sh; do
    [ -f "$REPO/$c" ] || ng "의존 파일 없음: $c"
  done
  say ""
  say "[데이터셋]  DATADIR=$DATADIR"
  D8="$DATADIR/8b"
  if [ -d "$D8" ]; then
    for f in c4-train.en_6_text_document.bin c4-train.en_6_text_document.idx \
             c4-validation-91205-samples.en_text_document.bin \
             c4-validation-91205-samples.en_text_document.idx \
             tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
      [ -s "$D8/$f" ] && ok "$f ($(du -h "$D8/$f" 2>/dev/null | cut -f1))" || ng "누락: $f"
    done
  else ng "$D8 없음 - bash $0 prep"; fi
  say "  여유공간 $(df -BG --output=avail "$DATADIR" 2>/dev/null | tail -1 | tr -d ' ')  (필요 ~90GB)"
  say ""
  say "[컨테이너]"
  if command -v docker >/dev/null 2>&1 && docker image inspect "$CONT" >/dev/null 2>&1; then
    ok "docker image $CONT"
  else ng "docker image $CONT 미확인 - bash $0 build 또는 CONT= 로 sqsh 경로 지정"; fi
  say ""
  say "[SLURM/GPU]"
  command -v sbatch >/dev/null 2>&1 && ok "slurm" || ng "sbatch 없음"
  srun --version >/dev/null 2>&1 && ok "srun" || ng "srun 없음"
  say "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  x $(nvidia-smi -L 2>/dev/null | wc -l)"
  say "  GPU MiB [$(mem | tr '\n' ' ')]   대기잡 $(squeue -h -u "$(whoami)" 2>/dev/null | wc -l)개"
  say ""
  if load_cfg "$CFG_BASE" 2>/dev/null; then CFG="$CFG_BASE"; say "[해석된 config]"; show_cfg; fi
  say ""
  say " 다음:  bash $0 dryrun  ->  bash $0 smoke  ->  bash $0 perf"
  ;;

build)
  say "컨테이너 빌드  |  로그 $V"
  cd "$REPO" || exit 1
  say "  base image: $(grep -m1 '^ARG FROM_IMAGE_NAME' Dockerfile | cut -d= -f2)"
  say "  nvcr.io 로그인이 안 되어 있으면 여기서 401 이 납니다:"
  say "    echo \$NGC_API_KEY | docker login nvcr.io -u '\$oauthtoken' --password-stdin"
  say ""
  docker build -t "$DOCKER_TAG" ${FROM_IMAGE_NAME:+--build-arg FROM_IMAGE_NAME=$FROM_IMAGE_NAME} \
    --build-arg GIT_COMMIT_ID="$(git rev-parse --short HEAD 2>/dev/null || echo na)" . 2>&1 | tee -a "$V" | tail -20
  if ! docker image inspect "$DOCKER_TAG" >/dev/null 2>&1; then
    ng "빌드 실패 - $V 확인"; exit 1
  fi
  ok "빌드 완료: $DOCKER_TAG"
  say ""
  say "sqsh 변환 (pyxis 는 로컬 docker 이미지를 못 읽습니다). 10~20분 소요."
  export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-$BASE/enroot-tmp}"
  mkdir -p "$ENROOT_TEMP_PATH"
  say "  ENROOT_TEMP_PATH=$ENROOT_TEMP_PATH  (여유 $(df -BG --output=avail "$ENROOT_TEMP_PATH" 2>/dev/null | tail -1 | tr -d ' '))"
  rm -f "$CONT"
  enroot import -o "$CONT" "dockerd://$DOCKER_TAG" 2>&1 | tee -a "$V" | tail -10
  if [ -s "$CONT" ]; then
    ok "sqsh 생성: $CONT ($(du -h "$CONT" | cut -f1))"
    say ""
    say " 다음 줄을 ~/.bashrc 에 추가하세요:"
    say "   export CONT=$CONT"
  else
    ng "sqsh 변환 실패 - $V 확인 (디스크 여유/도커 권한 확인)"
  fi
  ;;

prep)
  say "데이터셋 준비 (~85GB)  |  DATADIR=$DATADIR  로그 $V"
  mkdir -p "$DATADIR"
  cd "$REPO" || exit 1
  DATADIR="$DATADIR" bash data_scripts/download_8b.sh 2>&1 | tee -a "$V" | tail -20
  say ""
  ls -lh "$DATADIR/8b" 2>/dev/null | sed 's/^/   /'
  say ""
  say " 검증:  bash $0 check"
  ;;

dryrun)
  # hydra 로 resolved config 만 출력. GBS/GA/eval 주기를 실제 코드가 계산한 값으로 확인.
  launch "$CFG_BASE" "DRYRUN - resolved config 출력" \
    PRINT_CONFIG_ONLY=True USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 \
    MAX_STEPS=1 NCCL_TEST=0 WALLTIME_RUNANDTIME=20 WALLTIME=25 CHECK_COMPLIANCE=0
  say ""
  say " 로그에서 model.global_batch_size / trainer.val_check_interval 를 확인하세요."
  ;;

smoke)
  # 데이터셋 없이 mock dataset 으로 기능 확인
  S="${A2:-50}"
  launch "$CFG_BASE" "SMOKE - synthetic data ${S} step (기능 확인)" \
    USE_SYNTHETIC_DATA=1 VERIFY_MOUNTS=0 CHECK_COMPLIANCE=0 \
    MAX_STEPS="$S" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    NCCL_TEST=1 WALLTIME_RUNANDTIME=40 WALLTIME=45
  ;;

perf)
  S="${A2:-200}"
  # VAL_CHECK_INTERVAL(768) > MAX_STEPS 이므로 eval 이 돌지 않아 순수 train step time 만 측정됨
  launch "$CFG_BASE" "PERF - ${S} step 성능 측정 (eval 없음)" \
    MAX_STEPS="$S" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=60 WALLTIME=70
  ;;

fp8attn)
  S="${A2:-200}"
  launch "$CFG_ATTN" "PERF (FP8 attention) - ${S} step  <-- baseline 과 A/B 비교" \
    MAX_STEPS="$S" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
    CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=60 WALLTIME=70
  ;;

sweep)
  # MICRO_BATCH_SIZE 는 GBS 를 바꾸지 않는 유일한 성능 노브 (README_8b.md).
  # MINIBS=4 이므로 1/2/4 만 GA 가 정수.
  S="${A2:-150}"
  say "MICRO_BATCH_SIZE 스윕  (MINIBS=4 고정, GBS=16 불변)"
  BEST=""; BESTV=0
  for MBS in 1 2 4; do
    say ""
    say "########## MICRO_BATCH_SIZE = $MBS  (GA = $((4/MBS))) ##########"
    launch "$CFG_BASE" "SWEEP MBS=${MBS}" \
      MICRO_BATCH_SIZE="$MBS" MAX_STEPS="$S" LOG_EVERY_N_STEPS=1 LOG_METRICS=DELTA \
      CHECK_COMPLIANCE=0 WALLTIME_RUNANDTIME=50 WALLTIME=60 NCCL_TEST=0 || continue
    L=$(ls -1t "$LOGDIR"/*_0*.log 2>/dev/null | grep -v mountcheck | head -1)
    T=$(python3 -c "$PARSE_PY" "$L" "$NGPU" "421.59" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print('%.1f'%(d.get('tflops_per_gpu') or 0))")
    say " >>> MBS=$MBS  TFLOPS/GPU=$T"
    awk -v a="$T" -v b="$BESTV" 'BEGIN{exit !(a>b)}' && { BESTV="$T"; BEST="MBS=$MBS"; }
  done
  say ""
  say "=================================================="
  say " 최적: ${BEST:-측정실패}   TFLOPS/GPU ${BESTV}"
  say "=================================================="
  ;;

converge)
  # 실제 수렴 런. log_ppl < 3.3 도달까지. 4 GPU 에서는 매우 오래 걸립니다.
  say "수렴 런: TARGET_LOG_PPL=3.3 까지. 4 GPU(GBS=16)에서는 수십~수백 시간 규모입니다."
  say "중단해도 되도록 ENABLE_RERUNS/SHARE_RERUNS 로 체크포인트를 이어씁니다."
  launch "$CFG_BASE" "CONVERGE - log_ppl 3.3 목표" \
    ENABLE_RERUNS=1 SHARE_RERUNS=1 STORE_CKPTS_IN_LOGDIR=1 \
    WALLTIME_EXIT_MINUTES=10
  ;;

inspect)
  # 컨테이너를 열어 cuDNN / TransformerEngine 상태를 확인합니다.
  case "$CONT" in
    /*.sqsh|/*.squashfs) [ -s "$CONT" ] || { ng "CONT 없음: $CONT"; exit 1; } ;;
    dockerd://*|docker://*) ;;
    *) ng "CONT 형식이 잘못되었습니다: $CONT"; exit 1 ;;
  esac
  say "컨테이너 내부 점검  CONT=$CONT"
  say ""
  say "[호스트 환경 (컨테이너로 새어 들어감)]"
  for e in LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH CUDA_HOME CUDNN_PATH; do
    printf '   %-18s [%s]\n' "$e" "${!e:-unset}"
  done
  say ""
  say "[컨테이너 내부]"
  env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u CUDA_HOME -u CUDNN_PATH \
  srun --ntasks=1 -N1 --time=10 \
       --container-image="$CONT" \
       --no-container-mount-home --container-remap-root --container-writable \
       bash -c '
    echo "  LD_LIBRARY_PATH = [$LD_LIBRARY_PATH]"
    echo "  CUDNN_PATH      = [${CUDNN_PATH:-unset}]"
    echo "  CUDNN_VERSION   = [${CUDNN_VERSION:-unset}]"
    echo "  --- libcudnn 파일"
    ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so* 2>&1 | head -5
    echo "  --- dpkg"
    dpkg -l 2>/dev/null | grep -i cudnn || echo "    (cudnn 패키지 없음 - 빌드 실패)"
    echo "  --- ldconfig 캐시"
    ldconfig -p 2>/dev/null | grep -i cudnn | head -3 || echo "    (캐시에 없음)"
    echo "  --- import 테스트"
    python -c "import transformer_engine.pytorch as t; print(\"    TransformerEngine OK\")" \
      || echo "    TransformerEngine import 실패"
    python -c "import torch; print(\"    torch\", torch.__version__, \"cuda\", torch.version.cuda)"
  '
  say ""
  say " libcudnn 파일은 있는데 import 가 실패하면:"
  say "   CUDNN_LIBDIR=/usr/lib/x86_64-linux-gnu bash $0 smoke"
  say " dpkg 에 cudnn 이 아예 없으면 이미지를 다시 빌드해야 합니다."
  ;;

doctor)
  say "실행 환경 진단"
  say ""
  say "[SLURM]"
  command -v sbatch >/dev/null 2>&1 && ok "sbatch $(sbatch --version 2>/dev/null)" || ng "sbatch 없음"
  command -v sacct  >/dev/null 2>&1 && ok "sacct (accounting 사용 가능)" || ng "sacct 없음 - 종료상태 확인 불가"
  sinfo -h -o '   node=%n state=%t cpus=%c gres=%G' 2>/dev/null | sed 's/^/  /' || ng "sinfo 실패"
  say ""
  say "[pyxis / enroot]  run.sub 은 이 둘이 없으면 즉시 죽습니다"
  if srun --help 2>/dev/null | grep -q 'container-image'; then ok "pyxis 플러그인 (srun --container-image 인식)"
  else ng "pyxis 없음 - srun 에 --container-image 옵션이 없습니다"; fi
  command -v enroot >/dev/null 2>&1 && ok "enroot $(enroot version 2>/dev/null)" || ng "enroot 바이너리 없음"
  say ""
  say "[CONT]  현재값: $CONT"
  case "$CONT" in
    *.sqsh) [ -f "$CONT" ] && ok "sqsh 파일 존재 ($(du -h "$CONT" 2>/dev/null | cut -f1))" \
                           || ng "sqsh 경로가 존재하지 않습니다" ;;
    docker://*) ok "docker:// URI 형식 (pyxis 가 import 합니다. 첫 실행은 수 분 소요)" ;;
    *) ng "pyxis 는 로컬 docker 데몬 이미지를 직접 읽지 못합니다."
       say "      다음 중 하나로 바꾸세요:"
       say "        export CONT=docker://${CONT}"
       say "        또는 sqsh 로 변환:"
       say "        enroot import -o ${BASE}/llama31_8b.sqsh dockerd://${CONT}"
       say "        export CONT=${BASE}/llama31_8b.sqsh" ;;
  esac
  if command -v docker >/dev/null 2>&1; then
    docker image inspect "${CONT#docker://}" >/dev/null 2>&1 \
      && ok "docker 데몬에 ${CONT#docker://} 존재" || ng "docker 데몬에 해당 이미지 없음"
  fi
  say ""
  say "[GPU]"
  say "   nvidia-smi -L:"
  nvidia-smi -L 2>/dev/null | sed 's/^/     /'
  say "   보이는 GPU 수: $(nvidia-smi -L 2>/dev/null | wc -l)   요청 예정: 4"
  say ""
  say "[최근 잡 3건]"
  sacct -n -X -u "$(whoami)" -S "$(date -d '1 day ago' +%Y-%m-%d)" \
        --format=JobID%12,JobName%14,State%14,ExitCode%8,Elapsed%10,Reason%30 2>/dev/null \
    | tail -3 | sed 's/^/   /'
  say ""
  say "[최근 slurm stdout]"
  for f in $(ls -1t "$LOGDIR"/slurm-*.out "$REPO"/slurm-*.out 2>/dev/null | head -2); do
    say "   --- $f"
    tail -25 "$f" | sed 's/^/     /'
  done
  ;;

*) sed -n '3,26p' "$0" ;;
esac
