bin/bash: line 1: [: -ge: unary operator expected
[2026-08-31 21:41:53,216 utils.py:55 INFO RANK=0] Running command: /usr/bin/python3 /work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py --checkpoint-path /work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct --mlperf-accuracy-file /work/build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46/B300-SXM-270GBx4_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json --dataset-file /work/build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl --dtype int32
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt_tab.zip.
Traceback (most recent call last):
  File "/work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py", line 204, in <module>
    main()
  File "/work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py", line 150, in main
    targets, metrics = get_groundtruth(args.dataset_file)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py", line 97, in get_groundtruth
    data = pd.read_pickle(processed_dataset_file)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/pickle.py", line 185, in read_pickle
    with get_handle(
         ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/common.py", line 882, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/work/scripts/print_harness_result.py", line 195, in <module>
    all_acc_pass = print_session_results(log_dir)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scripts/print_harness_result.py", line 105, in print_session_results
    results = enumerate_results(base_dir)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scripts/print_harness_result.py", line 70, in enumerate_results
    acc_results = check_accuracy(wl)
                  ^^^^^^^^^^^^^^^^^^
  File "/work/code/common/mlcommons/accuracy_checker.py", line 958, in check_accuracy
    return accuracy_checker.get_accuracy()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/code/common/mlcommons/accuracy_checker.py", line 135, in get_accuracy
    output = self.run()
             ^^^^^^^^^^
  File "/work/code/common/mlcommons/accuracy_checker.py", line 124, in run
    return run_command(str(cmd), get_output=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/nvmitten/utils.py", line 76, in run_command
    raise subprocess.CalledProcessError(ret, cmd)
subprocess.CalledProcessError: Command '/usr/bin/python3 /work/3rdparty/mlc-inference/language/llama3.1-405b/evaluate-accuracy.py --checkpoint-path /work/build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct --mlperf-accuracy-file /work/build/logs/scaleout_B300-SXM-270GBx4_slurm-91_2026.08.31-18.59.46/B300-SXM-270GBx4_TRT/llama3.1-405b/Offline/mlperf_log_accuracy.json --dataset-file /work/build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl --dtype int32' returned non-zero exit status 1.
make: *** [Makefile:104: display_results] Error 1
srun: error: dell: task 0: Exited with exit code 2
dell@dell:/data/lsh/inference_results_v6.0/closed/NVIDIA$
[mlperf] 0:bash*                                                              
