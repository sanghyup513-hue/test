(mlperf) dell@dell:/work/build/logs/2026.07.16-20.55.24/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99/Offline$ cat mlperf_log_summary.txt
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 107.28
Tokens per second: 31313.2
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 4839236375
Max latency (ns)                : 2748992044494
Mean latency (ns)               : 1375799913383
50.00 percentile latency (ns)   : 1376001270594
90.00 percentile latency (ns)   : 2456551866221
95.00 percentile latency (ns)   : 2592397202342
97.00 percentile latency (ns)   : 2645816460436
99.00 percentile latency (ns)   : 2699717012989
99.90 percentile latency (ns)   : 2727242856485


================================================
Test Parameters Used
================================================
samples_per_query : 290400
target_qps : 110
ttft_latency (ns): 2000000000
tpot_latency (ns): 200000000
max_async_queries : 1
min_duration (ms): 2400000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 6023615788873153749
sample_index_rng_seed : 15036839855038426416
schedule_rng_seed : 9933818062894767841
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 24576
WARNING: sample_concatenate_permutation was set to true.
Generated samples per query might be different as the one in the setting.
Check the generated_samples_per_query line in the detailed log for the real
samples_per_query value

No warnings encountered during test.

No errors encountered during test.



(mlperf) dell@dell:/work/build/logs/2026.07.16-22.16.43/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99/Offline$ cat accuracy.txt

Results

{'rouge1': np.float64(44.5838), 'rouge2': np.float64(22.1557), 'rougeL': np.float64(28.7777), 'rougeLsum': np.float64(42.1594), 'gen_len': np.int64(28351264), 'gen_num': 24576, 'gen_tok_len': 7170515, 'tokens_per_sample': 291.8}
(mlperf) dell@dell:/work/build/logs/2026.07.16-22.16.43/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99/Offline$ cat mlperf_log_summary.txt

No warnings encountered during test.

No errors encountered during test.
