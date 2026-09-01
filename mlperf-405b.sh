#!/usr/bin/env bash
# =============================================================================
# MLPerf v6.0 — Llama3.1-405B  (TP2 x DP<N/2>)
#
#   bash mlperf-405b.sh clean                  잔여 잡/프로세스 정리
#   bash mlperf-405b.sh check                  자산 점검
#   bash mlperf-405b.sh prep                   데이터셋 + 모델 + 전처리
#
#   bash mlperf-405b.sh offline 4              Offline   (검증 완료: 1023.06, 104.8%)
#   bash mlperf-405b.sh server  4              Server
#   bash mlperf-405b.sh server  4 1.4          Server + QPS 지정
#   bash mlperf-405b.sh tune    4              Server QPS 자동탐색
#   bash mlperf-405b.sh accuracy 4             AccuracyOnly (ROUGE/exact_match)
#   bash mlperf-405b.sh result Server 4        최근 결과 재파싱
#
# 실행 호출은 2026-08-31 Offline 성공 케이스와 동일합니다.
#   --exclusive 유지 (CPU 344개 확보 — 제거 시 CPU 2개만 할당되어 로딩 불가)
#   CUDA_VISIBLE_DEVICES 는 export 하지 않음 (scaleout 자체 배치에 위임)
# GPU 배치는 감시만 하고, 3회 연속 위반 시에만 중단합니다.
#
# 공개값(8GPU): Offline 1951.6 / Server 1460.25 Tokens/s
# Server 제약 : TTFT p99 6.0s, TPOT p99 175ms
# 정확도 기준 : ROUGEL 21.8437 / exact_match 90.0418 / TOKENS_PER_SAMPLE 636.0
# =============================================================================
set -u
BASE="${BASE:-/data/lsh}"
REPO="$BASE/inference_results_v6.0/closed/NVIDIA"
SCRATCH="${MLPERF_SCRATCH_PATH:-$BASE/mlperf_inference_storage}"
SQSH=$(ls -1 "$REPO"/build/sqsh_images/*.sqsh 2>/dev/null | head -1)
MODEL="$SCRATCH/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct"
FP4="$SCRATCH/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4"
DATA="$SCRATCH/data/llama3.1-405b"
PREP="$SCRATCH/preprocessed_data/llama3.1-405b"
C2="$REPO/configs/B300-SXM-270GBx2"
PUB_OFF=1951.6; PUB_SRV=1460.25
TTFT_LIMIT=6000000000
TPOT_LIMIT=175000000
V="$BASE/405b-$(date +%m%d-%H%M).log"

MODE="${1:-}"; A2="${2:-}"; A3="${3:-}"
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_NTASKS SLURM_JOB_NODELIST
LAST_VALID=""; LAST_TPS=""

say(){ printf '%s\n' "$*"; }
ok(){ printf '   OK   %s\n' "$*"; }
ng(){ printf '   --   %s\n' "$*"; }
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

# ---------------------------------------------------------------------------
hard_clean(){
  scancel -u "$(whoami)" >>"$V" 2>&1; sleep 3
  pkill -9 -f 'trtllm-serve|trtllm-llmapi|code.main|run_scaleout' >>"$V" 2>&1; sleep 3
  local i pids
  for i in 1 2 3 4 5 6; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
  # CG(completing) 상태가 빠질 때까지 대기
  for i in 1 2 3 4 5 6; do
    squeue -h -u "$(whoami)" | grep -q . || break
    sleep 5
  done
  sinfo -h -o '%t' | grep -qiE 'drain|down' && \
    sudo scontrol update NodeName="$(hostname -s)" State=RESUME >>"$V" 2>&1
  local busy; busy=$(mem | awk '$1>1000{n++}END{print n+0}')
  if [ "$busy" -gt 0 ]; then
    ng "GPU 메모리 미해제: [$(mem | tr '\n' ' ')]"
    ng "확인: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    return 1
  fi
  ok "GPU 전부 비움 [$(mem | tr '\n' ' ')]   대기잡 $(squeue -h -u "$(whoami)" | wc -l)개"
  return 0
}

# ---------------------------------------------------------------------------
mkcfg(){   # $1 = GPU 수.  x2(atomic, TP2 1인스턴스) -> xN(harness), DP = N/2
  local n="$1" dp=$(( $1 / 2 )) cN="$REPO/configs/B300-SXM-270GBx$1" oq sq
  oq=$(awk -v d="$dp" 'BEGIN{printf "%.2f", 0.75*d}')
  sq=$(awk -v d="$dp" 'BEGIN{printf "%.2f", 0.80*d}')
  mkdir -p "$cN/Offline" "$cN/Server"
  sed "s/offline_expected_qps: 0.75/offline_expected_qps: $oq/" \
      "$C2/Offline/llama3_1-405b.py" > "$cN/Offline/llama3_1-405b.py"
  sed "s/server_target_qps: 0.8/server_target_qps: $sq/" \
      "$C2/Server/llama3_1-405b.py"  > "$cN/Server/llama3_1-405b.py"
}
setsrv(){
  sed -i -E "s/server_target_qps: [0-9.]+/server_target_qps: $2/" \
    "$REPO/configs/B300-SXM-270GBx$1/Server/llama3_1-405b.py"
  grep -h 'server_target_qps' "$REPO/configs/B300-SXM-270GBx$1/Server/llama3_1-405b.py" | sed 's/^/        /'
}

# ---------------------------------------------------------------------------
report(){   # $1=Offline|Server|Accuracy  $2=GPU수
  local scen="$1" n="$2" D S TPS VAL PCT TGT TTFT TPOT LATOK
  D=$(ls -1dt "$REPO"/build/logs/scaleout_* 2>/dev/null | head -1)
  S=$(find "$D" -name mlperf_log_summary.txt 2>/dev/null | head -1)
  if [ "$scen" = "Server" ]; then
    TGT=$(awk -v p="$PUB_SRV" -v n="$n" 'BEGIN{printf "%.0f", p*n/8}')
  else
    TGT=$(awk -v p="$PUB_OFF" -v n="$n" 'BEGIN{printf "%.0f", p*n/8}')
  fi
  TPS=""; VAL=""; PCT=""; TTFT=""; TPOT=""
  if [ -n "$S" ]; then
    VAL=$(grep -m1 'Result is' "$S" | awk -F': *' '{print $2}')
    TPS=$(grep -m1 -E 'Completed tokens per second|Tokens per second' "$S" \
          | grep -oE '[0-9]+\.?[0-9]*' | head -1)
    TTFT=$(grep -i 'ttft' "$S" | grep '99.00' | grep -oE '[0-9]{6,}' | tail -1)
    TPOT=$(grep -i 'tpot' "$S" | grep '99.00' | grep -oE '[0-9]{6,}' | tail -1)
  fi
  [ -n "$TPS" ] && PCT=$(awk -v a="$TPS" -v b="$TGT" 'BEGIN{printf "%.1f",a/b*100}')

  say ""
  say "=================================================="
  say " llama3.1-405b ${scen}   GPU ${n}장 (TP2 x DP$((n/2)))"
  say "=================================================="
  printf " %-10s %s\n" "GPU별MiB" "$(mem | tr '\n' ' ')"
  printf " %-10s %s\n" "TOKENS/S" "${TPS:-측정실패}"
  printf " %-10s %s\n" "TARGET"   "$TGT  (공개값 x ${n}/8)"
  printf " %-10s %s\n" "달성률"    "${PCT:-0}%"
  printf " %-10s %s\n" "LOADGEN"  "${VAL:-N/A}"
  if [ "$scen" = "Server" ]; then
    LATOK=1
    if [ -n "$TTFT" ]; then
      awk -v a="$TTFT" -v b="$TTFT_LIMIT" \
        'BEGIN{printf "  TTFT p99 : %.2f s   (한도 6.00 s)   %s\n", a/1e9, (a<=b?"OK":"초과")}'
      [ "$TTFT" -gt "$TTFT_LIMIT" ] && LATOK=0
    else say "  TTFT p99 : 요약에 없음"; fi
    if [ -n "$TPOT" ]; then
      awk -v a="$TPOT" -v b="$TPOT_LIMIT" \
        'BEGIN{printf "  TPOT p99 : %.1f ms  (한도 175.0 ms)  %s\n", a/1e6, (a<=b?"OK":"초과")}'
      [ "$TPOT" -gt "$TPOT_LIMIT" ] && LATOK=0
    else say "  TPOT p99 : 요약에 없음"; fi
    printf " %-10s %s\n" "지연제약" "$([ "$LATOK" = 1 ] && echo 충족 || echo 위반)"
  fi
  say "=================================================="
  say " CODE: V6.L405.G${n}.${scen}.${TPS:-0}.${PCT:-0}.${VAL:-FAIL}"
  say "=================================================="
  say " 로그: $V"
  LAST_VALID="$VAL"; LAST_TPS="$TPS"
}

# ---------------------------------------------------------------------------
# Offline 성공 케이스와 동일한 호출. --exclusive 유지, 환경변수 주입 없음.
# ---------------------------------------------------------------------------
launch(){   # $1=Offline|Server  $2=GPU수  $3=추가run-args  $4=제한시간
  local scen="$1" n="$2" extra="$3" wall="$4" P t=0 viol=0 bad
  local cN="$REPO/configs/B300-SXM-270GBx$n"

  [ -f "$SQSH" ] || { ng "sqsh 없음"; return 1; }
  [ -f "$PREP/input_ids_padded.npy" ] || { ng "전처리 미완 — bash $0 prep"; return 1; }
  [ $(( n % 2 )) -eq 0 ] || { ng "GPU 수는 짝수여야 합니다 (TP2 구조)"; return 1; }
  [ -f "$cN/$scen/llama3_1-405b.py" ] || mkcfg "$n"

  hard_clean || return 1
  export SLURM_MPI_TYPE=pmi2
  cd "$REPO" || return 1

  say ""
  say "$scen 실행  |  GPU ${n}장 (TP2 x DP$((n/2)))  |  405B 로딩 15~30분"
  say "60초마다 GPU 메모리를 표시합니다. 앞 ${n}장만 차면 정상입니다."

  salloc --nodes=1 --gres=gpu:"$n" --exclusive --time="$wall" \
    ./scaleout/run_scaleout.sh \
      --stage all \
      --atomic-system B300-SXM-270GBx2 \
      --gpus-per-node "$n" --dp-multiplicity $(( n / 2 )) \
      --container-image "$SQSH" \
      --mlperf-scratch-path "$SCRATCH" \
      --extra-srun-flags "--overlap --cpu-bind=none --mpi=pmi2" \
      --server-spawn-time 180 \
      --run-args "--benchmarks=llama3.1-405b --scenarios=$scen --core_type=trtllm_endpoint $extra --readiness_timeout=3600" \
    >>"$V" 2>&1 &
  P=$!

  while kill -0 $P 2>/dev/null; do
    sleep 60; t=$((t+1))
    say "   $(date +%H:%M)  GPU별MiB [$(mem | tr '\n' ' ')]"
    # 범위 밖 GPU 사용 감시 — 5분 경과 후, 3회 연속 위반 시에만 중단
    if [ "$t" -ge 5 ]; then
      bad=$(mem | awk -v n="$n" 'NR>n && $1>50000{c++}END{print c+0}')
      if [ "$bad" -gt 0 ]; then
        viol=$((viol+1))
        say "        경고: GPU ${n}번 이후 ${bad}장 사용 중 (${viol}/3)"
        if [ "$viol" -ge 3 ]; then
          ng "3회 연속 위반 — 배치 실패로 중단합니다."
          kill $P 2>/dev/null; hard_clean
          grep -E 'Harness system|Atomic system|GPUs per node|DP multiplicity|Total GPUs' "$V" \
            | tail -6 | sed 's/^/   /'
          return 2
        fi
      else viol=0; fi
    fi
  done
  wait $P
  return 0
}

# =============================================================================
case "$MODE" in

clean) hard_clean; squeue -u "$(whoami)" ;;

result) report "${A2:-Offline}" "${A3:-4}" ;;

check)
  say "Llama3.1-405B 사전점검"
  say ""
  say "[용량] scratch 여유 $(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -d ' ')"
  say "[자산]"
  [ -d "$FP4" ] && ok "FP4 ($(du -sh "$FP4" 2>/dev/null|cut -f1), safetensors $(ls "$FP4"/*.safetensors 2>/dev/null|wc -l)개)" || ng "FP4 없음"
  [ -d "$MODEL" ] && ok "토크나이저 ($(du -sh "$MODEL" 2>/dev/null|cut -f1))" || ng "토크나이저 없음"
  ls "$DATA"/*.pkl >/dev/null 2>&1 && ok "데이터셋 $(ls "$DATA"/*.pkl|wc -l)개" || ng "데이터셋 없음"
  [ -f "$PREP/input_ids_padded.npy" ] && ok "전처리 완료" || ng "전처리 미완"
  say ""
  say "[repo 패치 상태]"
  [ "$(grep -c 'versioning.parse(C.VERSION' "$REPO/code/common/mlcommons/loadgen.py" 2>/dev/null)" -ge 1 ] \
    && ok "loadgen 버전고정" || ng "loadgen 버전고정 누락"
  [ "$(grep -c 'mpi_mode=legacy' "$REPO/scaleout/run_scaleout.sh" 2>/dev/null)" -ge 1 ] \
    && ok "mpi_mode=legacy" || ng "mpi_mode=legacy 누락"
  [ "$(grep -c "locals().get('gpu_ids'" "$REPO/code/llmlib/launch_server.py" 2>/dev/null)" -ge 1 ] \
    && ok "gpu_ids 패치" || ng "gpu_ids 패치 누락"
  say ""
  say "[GPU 현황] $(mem | tr '\n' ' ')     [대기잡] $(squeue -h -u "$(whoami)" | wc -l)개"
  say ""
  say "[config]"
  for n in 4 6; do
    [ -d "$REPO/configs/B300-SXM-270GBx$n" ] && \
      grep -h 'expected_qps\|server_target_qps' "$REPO/configs/B300-SXM-270GBx$n"/*/llama3_1-405b.py 2>/dev/null | sed "s/^/   x$n: /"
  done
  say ""
  say " 실행:  bash $0 offline 4  |  bash $0 server 4  |  bash $0 tune 4  |  bash $0 accuracy 4"
  ;;

prep)
  say "자산 준비  |  로그 $V"
  mkdir -p "$DATA" "$PREP"
  R2="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
  cd "$DATA" || exit 1
  if ls "$DATA"/*8313* >/dev/null 2>&1; then ok "데이터셋 이미 존재"; else
    bash <(curl -s "$R2") https://inference.mlcommons-storage.org/metadata/llama3-1-405b-dataset-8313.uri 2>&1 | tail -3
    bash <(curl -s "$R2") https://inference.mlcommons-storage.org/metadata/llama3-1-405b-calibration-dataset-512.uri 2>&1 | tail -3
    find "$DATA" -mindepth 2 -name '*.pkl' -exec mv -n {} "$DATA/" \; 2>/dev/null
  fi
  ls -lh "$DATA"/*.pkl 2>/dev/null | sed 's/^/   /'
  [ -d "$FP4" ] || hf download nvidia/Llama-3.1-405B-Instruct-FP4 --local-dir "$FP4"
  cat > /tmp/prep405.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=405b-prep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=$BASE/405b-prep-%j.log
srun --container-image $SQSH \\
     --container-mounts $REPO:/work,$SCRATCH:/home/mlperf_inference_storage \\
     --container-workdir /work --container-remap-root \\
     --export ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \\
     bash -lc 'set -x; make link_dirs; python3 -u code/llama3_1-405b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed_data'
EOF
  J=$(sbatch --parsable /tmp/prep405.sbatch)
  say "   전처리 JobID $J   tail -f $BASE/405b-prep-${J}.log"
  ;;

offline)
  N="${A2:-4}"
  say "Llama3.1-405B Offline  |  GPU ${N}장"
  launch Offline "$N" "" "04:00:00" && report Offline "$N"
  ;;

server)
  N="${A2:-4}"
  [ -f "$REPO/configs/B300-SXM-270GBx$N/Server/llama3_1-405b.py" ] || mkcfg "$N"
  [ -n "$A3" ] && { say "server_target_qps -> $A3"; setsrv "$N" "$A3"; }
  say "Llama3.1-405B Server  |  GPU ${N}장"
  say "지연 제약: TTFT p99 6.0s / TPOT p99 175ms — 위반 시 INVALID"
  launch Server "$N" "" "03:00:00" && report Server "$N"
  ;;

tune)
  N="${A2:-4}"
  say "Server QPS 자동탐색  |  GPU ${N}장"
  [ -f "$REPO/configs/B300-SXM-270GBx$N/Server/llama3_1-405b.py" ] || mkcfg "$N"
  BASEQ=$(awk -v d="$((N/2))" 'BEGIN{printf "%.2f", 0.80*d}')
  BEST=""
  for f in 1.00 0.875 0.75 0.625; do
    q=$(awk -v b="$BASEQ" -v f="$f" 'BEGIN{printf "%.2f", b*f}')
    say ""
    say "########## server_target_qps = $q ##########"
    setsrv "$N" "$q"
    launch Server "$N" "" "03:00:00" || break
    report Server "$N"
    if [ "$LAST_VALID" = "VALID" ]; then
      BEST="QPS $q  →  ${LAST_TPS:-?} Tokens/s"
      say ""; say " >>> VALID. 탐색 종료."; break
    fi
    say " >>> INVALID. QPS 를 낮춰 재시도합니다."
  done
  say ""
  say "=================================================="
  say " 탐색 결과: ${BEST:-모두 INVALID}"
  say "=================================================="
  ;;

accuracy)
  N="${A2:-4}"
  say "Llama3.1-405B AccuracyOnly (Offline)  |  GPU ${N}장"
  # 채점기 필수 자산 사전 검증 (없으면 추론 3시간 후 채점 단계에서 실패)
  AF=0
  for f in "$MODEL/config.json" "$MODEL/tokenizer.json" \
           "$PREP/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"; do
    if [ -f "$f" ] && [ -s "$f" ]; then ok "$(basename "$f")"
    else ng "누락: $f"; AF=1; fi
  done
  if [ "$AF" -eq 1 ]; then
    say ""
    say " 채점 자산이 없어 중단합니다. 아래를 먼저 실행하세요:"
    say "   hf download meta-llama/Llama-3.1-405B-Instruct config.json tokenizer.json --local-dir $MODEL"
    say "   cp $DATA/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl $PREP/"
    exit 1
  fi
  say "기준: ROUGEL 21.8437 / exact_match 90.0418 / TOKENS_PER_SAMPLE 636.0"
  launch Offline "$N" "--test_mode=AccuracyOnly" "05:00:00"
  D=$(ls -1dt "$REPO"/build/logs/scaleout_* | head -1)
  say ""
  say "=================================================="
  say " 정확도 결과"
  say "=================================================="
  # run_harness 가 display_results 로 자동 채점 → stdout.txt 에 기록됨
  if [ -f "$D/stdout.txt" ]; then
    grep -iE 'rouge|exact_match|tokens_per_sample|accuracy|PASSED|FAILED' "$D/stdout.txt" \
      | tail -15 | sed 's/^/   /'
  else
    ng "stdout.txt 없음"
  fi
  find "$D" -name 'accuracy*.txt' -o -name 'mlperf_log_accuracy.json' 2>/dev/null | sed 's/^/   파일: /'
  say "=================================================="
  ;;

*) sed -n '4,22p' "$0" ;;
esac
