cd ~
cp mlperf-405b.sh mlperf-405b.sh.bak

python3 - <<'EOF'
p='mlperf-405b.sh'
s=open(p).read()
old='''accuracy)
  N="${A2:-4}"
  say "Llama3.1-405B AccuracyOnly (Offline)  |  GPU ${N}장"'''
new='''accuracy)
  N="${A2:-4}"
  say "Llama3.1-405B AccuracyOnly (Offline)  |  GPU ${N}장"
  # 채점기 필수 자산 사전 검증 (없으면 추론 3시간 후 채점 단계에서 실패)
  AF=0
  for f in "$MODEL/config.json" "$MODEL/tokenizer.json" \\
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
  fi'''
assert old in s, "패턴 불일치"
open(p,'w').write(s.replace(old,new))
print("사전검증 추가 완료")
EOF

bash -n mlperf-405b.sh && echo "문법 OK"
