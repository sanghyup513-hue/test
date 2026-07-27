CPU	SPEC CPU 2017	"* sysbench로 20000 까지 소수 계산
* stress-ng로 10분동안 cpu 부하"	"CPU: Intel(R) Xeon(R) Platinum 8462Y+
코어수: 128 | NUMA 노드: 2 | 결과: /opt/results/cpu_20260722_153610
===== [single] threads=1  =====
  -> events/sec: 1188.86
===== [multi] threads=128  =====
  -> events/sec: 76058.17
===== [numa0] threads=64 (via numactl --cpunodebind=0 --membind=0) =====
  -> events/sec: 38224.68
===== [numa1] threads=64 (via numactl --cpunodebind=1 --membind=1) =====
  -> events/sec: 39334.84

----- NUMA 노드 간 편차 -----
  numa0: 38,225 ev/s (97.2%)
  numa1: 39,335 ev/s (100.0%)

----- 멀티코어 스케일링 효율 -----
  single: 1,189 ev/s | multi: 76,058 ev/s | 128 cores
  스케일링 효율: 50.0%

===== 완료. 상세: /opt/results/cpu_20260722_153610 ====="
<img width="1337" height="82" alt="image" src="https://github.com/user-attachments/assets/a5bd6d8d-e2b0-461d-bf24-c833fa78d829" />



Mem	Host Memory Bandwidth test	"* stream으로 19GB Copy/Scale/Add/Triad 연산을 20번씩 반복
* stress-ng로 10분동안 메모리 부하"	"NUMA 노드: 2 | ARRAY_SIZE=800000000 | NTIMES=20 | 결과: /opt/results/mem_260724_1037
node 0 size: 515520 MB
node 1 size: 516018 MB
===== [local_node0] numactl --cpunodebind=0 --membind=0 =====
  -> Copy:    191038.9 MB/s (191.0 GB/s)
  -> Scale:   133337.7 MB/s (133.3 GB/s)
  -> Add:     155317.0 MB/s (155.3 GB/s)
  -> Triad:   151312.2 MB/s (151.3 GB/s)
===== [local_node1] numactl --cpunodebind=1 --membind=1 =====
  -> Copy:    182862.3 MB/s (182.9 GB/s)
  -> Scale:   139129.0 MB/s (139.1 GB/s)
  -> Add:     166953.7 MB/s (167.0 GB/s)
  -> Triad:   173408.7 MB/s (173.4 GB/s)

----- 노드별 로컬 대역폭 (Triad) -----
  local_node0: 151.3 GB/s (87.3%)
  local_node1: 173.4 GB/s (100.0%)
===== [remote_cpu0_mem1] numactl --cpunodebind=0 --membind=1 =====
  -> Copy:    116366.8 MB/s (116.4 GB/s)
  -> Scale:    87672.4 MB/s (87.7 GB/s)
  -> Add:     101623.3 MB/s (101.6 GB/s)
  -> Triad:   101053.2 MB/s (101.1 GB/s)
===== [remote_cpu1_mem0] numactl --cpunodebind=1 --membind=0 =====
  -> Copy:    116410.2 MB/s (116.4 GB/s)
  -> Scale:    90780.1 MB/s (90.8 GB/s)
  -> Add:     101194.3 MB/s (101.2 GB/s)
  -> Triad:   101585.2 MB/s (101.6 GB/s)
===== [full_system]  =====
  -> Copy:    166236.0 MB/s (166.2 GB/s)
  -> Scale:   137633.1 MB/s (137.6 GB/s)
  -> Add:     161615.9 MB/s (161.6 GB/s)
  -> Triad:   160153.5 MB/s (160.2 GB/s)
===== [stress] stress-ng vm 600s (메모리 패턴 부하 + ECC 전후 비교) =====
  ECC 카운터: ecc_before.txt / ecc_after.txt (diff로 부하 중 증가분 확인)

===== 완료. 상세: /opt/results/mem_260724_1037 ====="
<img width="1337" height="84" alt="image" src="https://github.com/user-attachments/assets/e81dd3f7-420e-467e-81c0-e2b529e45a10" />


"nvbandwidth bandwidth 
(H2D & D2H)"	"* nvbandwidth 로 대량의 데이터를 memcpy(복사)해서 측정
* 같은 테스트를 5번 반복해서 안정적인 평균값 산출"	"nvbandwidth Version: v0.10.0
Built from Git version: v0.10.0

CUDA Runtime Version: 13.2 (13020)
CUDA Driver Version (API): 13.2 (13020)
Driver Version: 595.71.05

Running device_to_device_latency_sm.
Device to Device Latency SM GPU(row) <-> GPU(column) (ns)
           0         1         2         3         4         5         6         7
 0       N/A    826.42    827.77    826.47    827.16    829.51    822.95    828.79
 1    825.83       N/A    828.76    829.45    827.61    827.40    824.86    829.52
 2    826.10    828.28       N/A    825.82    828.50    828.32    825.36    827.22
 3    825.45    829.52    826.73       N/A    823.29    826.17    825.92    822.68
 4    825.63    827.14    828.62    822.42       N/A    825.25    826.08    823.46
 5    828.18    827.33    828.85    825.74    825.07       N/A    829.36    826.05
 6    822.02    825.19    825.91    825.90    826.67    829.24       N/A    828.40
 7    827.00    829.47    827.23    821.94    824.07    825.80    827.99       N/A

SUM device_to_device_latency_sm 46287.84

NOTE: The reported results may not reflect the full capabilities of the platform.
Performance can vary with software drivers, hardware clocks, and system topology.



#####################################################

nvbandwidth Version: v0.10.0
Built from Git version: v0.10.0

CUDA Runtime Version: 13.2 (13020)
CUDA Driver Version (API): 13.2 (13020)
Driver Version: 595.71.05

Running device_to_device_memcpy_read_ce.
memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0       N/A    395.08    395.08    395.08    395.14    395.19    395.12    395.12
 1    395.08       N/A    395.08    395.08    395.08    395.12    395.18    395.08
 2    395.08    395.08       N/A    395.08    395.16    395.18    395.07    395.19
 3    395.08    395.08    395.23       N/A    395.23    395.18    395.19    395.19
 4    395.19    395.08    395.08    395.08       N/A    395.08    395.25    395.08
 5    395.08    395.08    395.08    395.08    395.08       N/A    395.08    395.08
 6    395.08    395.08    395.08    395.08    395.08    395.08       N/A    395.08
 7    395.05    395.18    395.19    395.18    395.08    395.23    395.08       N/A

SUM device_to_device_memcpy_read_ce 22126.64
COEFFICIENT_OF_VARIATION device_to_device_memcpy_read_ce 0.00

NOTE: The reported results may not reflect the full capabilities of the platform.
Performance can vary with software drivers, hardware clocks, and system topology.


#####################################################

nvbandwidth Version: v0.10.0
Built from Git version: v0.10.0

CUDA Runtime Version: 13.2 (13020)
CUDA Driver Version (API): 13.2 (13020)
Driver Version: 595.71.05

Running device_to_device_memcpy_write_ce.
memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0       N/A    397.44    397.48    397.48    397.48    397.44    397.50    397.44
 1    397.50       N/A    397.48    397.48    397.48    397.46    397.48    397.48
 2    397.48    397.48       N/A    397.44    397.48    397.46    397.52    397.52
 3    397.48    397.48    397.48       N/A    397.48    397.46    397.48    397.46
 4    397.46    397.48    397.46    397.46       N/A    397.46    397.44    397.46
 5    397.44    397.50    397.46    397.48    397.48       N/A    397.46    397.46
 6    397.46    397.50    397.48    397.44    397.50    397.48       N/A    397.44
 7    397.44    397.48    397.46    397.44    397.44    397.48    397.46       N/A

SUM device_to_device_memcpy_write_ce 22258.35
COEFFICIENT_OF_VARIATION device_to_device_memcpy_write_ce 0.00

NOTE: The reported results may not reflect the full capabilities of the platform.
Performance can vary with software drivers, hardware clocks, and system topology.



#####################################################

nvbandwidth Version: v0.10.0
Built from Git version: v0.10.0

CUDA Runtime Version: 13.2 (13020)
CUDA Driver Version (API): 13.2 (13020)
Driver Version: 595.71.05

Running device_to_host_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0     55.20     55.20     55.18     55.18     55.25     55.25     55.26     55.20

SUM device_to_host_memcpy_ce 441.72
COEFFICIENT_OF_VARIATION device_to_host_memcpy_ce 0.00

NOTE: The reported results may not reflect the full capabilities of the platform.
Performance can vary with software drivers, hardware clocks, and system topology.



#####################################################


nvbandwidth Version: v0.10.0
Built from Git version: v0.10.0

CUDA Runtime Version: 13.2 (13020)
CUDA Driver Version (API): 13.2 (13020)
Driver Version: 595.71.05

Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0     55.37     55.36     55.37     55.36     55.36     55.36     55.36     55.33

SUM host_to_device_memcpy_ce 442.87
COEFFICIENT_OF_VARIATION host_to_device_memcpy_ce 0.00

NOTE: The reported results may not reflect the full capabilities of the platform.
Performance can vary with software drivers, hardware clocks, and system topology."
<img width="1270" height="162" alt="image" src="https://github.com/user-attachments/assets/d8d3ee2c-6001-4958-b40e-97d00b1403aa" />





"p2pBandwidthLatencyTest
(CudaSamples, bidirectional)"	CudaSamples를 활용하여 모든 GPU 쌍(pair)에 대해 직접 메모리를 복사	"[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA H100 80GB HBM3, pciBusID: 19, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA H100 80GB HBM3, pciBusID: 3b, pciDeviceID: 0, pciDomainID:0
Device: 2, NVIDIA H100 80GB HBM3, pciBusID: 4c, pciDeviceID: 0, pciDomainID:0
Device: 3, NVIDIA H100 80GB HBM3, pciBusID: 5d, pciDeviceID: 0, pciDomainID:0
Device: 4, NVIDIA H100 80GB HBM3, pciBusID: 9b, pciDeviceID: 0, pciDomainID:0
Device: 5, NVIDIA H100 80GB HBM3, pciBusID: bb, pciDeviceID: 0, pciDomainID:0
Device: 6, NVIDIA H100 80GB HBM3, pciBusID: cb, pciDeviceID: 0, pciDomainID:0
Device: 7, NVIDIA H100 80GB HBM3, pciBusID: db, pciDeviceID: 0, pciDomainID:0
Device=0 CAN Access Peer Device=1
Device=0 CAN Access Peer Device=2
Device=0 CAN Access Peer Device=3
Device=0 CAN Access Peer Device=4
Device=0 CAN Access Peer Device=5
Device=0 CAN Access Peer Device=6
Device=0 CAN Access Peer Device=7
Device=1 CAN Access Peer Device=0
Device=1 CAN Access Peer Device=2
Device=1 CAN Access Peer Device=3
Device=1 CAN Access Peer Device=4
Device=1 CAN Access Peer Device=5
Device=1 CAN Access Peer Device=6
Device=1 CAN Access Peer Device=7
Device=2 CAN Access Peer Device=0
Device=2 CAN Access Peer Device=1
Device=2 CAN Access Peer Device=3
Device=2 CAN Access Peer Device=4
Device=2 CAN Access Peer Device=5
Device=2 CAN Access Peer Device=6
Device=2 CAN Access Peer Device=7
Device=3 CAN Access Peer Device=0
Device=3 CAN Access Peer Device=1
Device=3 CAN Access Peer Device=2
Device=3 CAN Access Peer Device=4
Device=3 CAN Access Peer Device=5
Device=3 CAN Access Peer Device=6
Device=3 CAN Access Peer Device=7
Device=4 CAN Access Peer Device=0
Device=4 CAN Access Peer Device=1
Device=4 CAN Access Peer Device=2
Device=4 CAN Access Peer Device=3
Device=4 CAN Access Peer Device=5
Device=4 CAN Access Peer Device=6
Device=4 CAN Access Peer Device=7
Device=5 CAN Access Peer Device=0
Device=5 CAN Access Peer Device=1
Device=5 CAN Access Peer Device=2
Device=5 CAN Access Peer Device=3
Device=5 CAN Access Peer Device=4
Device=5 CAN Access Peer Device=6
Device=5 CAN Access Peer Device=7
Device=6 CAN Access Peer Device=0
Device=6 CAN Access Peer Device=1
Device=6 CAN Access Peer Device=2
Device=6 CAN Access Peer Device=3
Device=6 CAN Access Peer Device=4
Device=6 CAN Access Peer Device=5
Device=6 CAN Access Peer Device=7
Device=7 CAN Access Peer Device=0
Device=7 CAN Access Peer Device=1
Device=7 CAN Access Peer Device=2
Device=7 CAN Access Peer Device=3
Device=7 CAN Access Peer Device=4
Device=7 CAN Access Peer Device=5
Device=7 CAN Access Peer Device=6

***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

P2P Connectivity Matrix
     D\D     0     1     2     3     4     5     6     7
     0       1     1     1     1     1     1     1     1
     1       1     1     1     1     1     1     1     1
     2       1     1     1     1     1     1     1     1
     3       1     1     1     1     1     1     1     1
     4       1     1     1     1     1     1     1     1
     5       1     1     1     1     1     1     1     1
     6       1     1     1     1     1     1     1     1
     7       1     1     1     1     1     1     1     1
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 2841.07  37.53  37.82  36.74  37.73  36.87  36.75  36.99
     1  37.11 2888.00  37.52  37.20  37.76  37.32  37.18  37.56
     2  37.23  37.49 2879.19  36.85  37.24  37.31  36.96  37.55
     3  36.70  37.45  37.13 2876.37  37.08  36.96  37.33  37.85
     4  37.07  38.27  37.62  37.28 2877.53  37.91  37.32  37.29
     5  38.04  37.56  37.93  37.39  37.57 2883.67  37.41  37.96
     6  38.02  37.50  37.26  37.11  38.30  37.18 2871.25  37.42
     7  37.22  37.55  38.40  38.15  38.11  37.70  37.28 2884.50
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 2862.54 365.30 375.46 375.92 376.19 376.40 376.57 375.09
     1 375.88 2897.71 376.35 376.77 376.15 375.95 375.29 376.91
     2 376.67 393.28 2887.50 376.53 376.01 376.24 376.19 375.17
     3 375.74 376.36 393.07 2891.01 393.10 393.01 393.10 393.34
     4 392.01 393.17 392.44 392.83 2898.05 392.92 392.90 392.85
     5 392.95 393.78 393.24 393.72 393.51 2900.91 392.96 392.68
     6 393.25 392.38 393.56 392.91 392.56 393.61 2899.56 393.11
     7 392.37 392.22 392.85 393.15 393.13 393.56 393.08 2895.53
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 2947.68  45.75  45.61  45.57  52.37  52.55  52.19  52.33
     1  45.47 2958.49  45.76  45.37  52.74  52.75  52.58  52.91
     2  45.83  46.03 2951.33  45.48  52.55  52.68  51.86  53.03
     3  45.39  45.79  45.25 2947.50  51.66  52.34  52.14  52.86
     4  52.48  53.03  53.02  51.89 2951.59  52.85  53.30  52.47
     5  52.71  52.88  52.32  52.01  52.65 2945.42  52.41  52.62
     6  52.43  52.36  52.05  52.26  52.49  52.48 2949.24  52.16
     7  51.73  52.18  53.01  53.16  52.77  53.28  52.56 2951.16
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 2946.72 739.22 737.95 742.06 741.30 740.12 741.32 740.91
     1 774.45 2947.68 774.37 774.99 775.01 773.53 774.79 774.68
     2 775.34 773.47 2938.76 774.41 774.28 773.57 774.29 773.75
     3 774.16 774.76 773.23 2943.43 774.79 773.38 774.69 774.52
     4 774.04 774.45 773.22 772.63 2946.90 773.83 773.59 774.40
     5 775.16 773.87 773.44 774.89 772.40 2950.81 775.03 774.34
     6 774.15 774.20 774.61 775.00 774.27 773.68 2947.42 774.79
     7 773.97 773.49 774.25 774.19 774.26 774.67 774.59 2945.94
P2P=Disabled Latency Matrix (us)
   GPU     0      1      2      3      4      5      6      7
     0   2.03  18.06  19.18  18.46  16.64  18.80  15.55  16.34
     1  18.49   2.04  18.25  17.81  16.44  18.11  16.20  17.24
     2  17.13  18.98   2.15  17.63  16.21  18.48  16.38  15.97
     3  18.27  18.16  15.67   2.00  15.62  18.32  16.48  15.84
     4  19.23  16.10  15.77  14.57   1.96  16.54  14.13  14.71
     5  18.29  19.21  16.14  18.12  16.48   2.22  15.97  15.72
     6  18.05  15.82  14.72  18.32  15.72  16.47   1.99  16.16
     7  16.17  16.05  16.21  18.68  16.70  16.84  16.46   2.03

   CPU     0      1      2      3      4      5      6      7
     0   2.25   7.34   7.22   7.10   6.84   6.79   6.77   6.72
     1   7.21   2.19   7.15   7.98   6.74   6.73   6.66   6.77
     2   7.24   7.21   2.18   7.16   6.75   6.72   6.69   6.63
     3   7.11   7.28   7.16   2.25   6.73   6.70   6.66   6.60
     4   6.79   6.75   6.83   6.80   2.10   6.39   6.34   6.31
     5   6.71   6.70   6.87   6.82   6.48   2.09   6.35   6.34
     6   6.68   6.73   6.72   6.75   6.38   6.37   2.01   6.34
     7   6.77   6.70   6.72   6.73   6.44   6.48   6.41   2.02
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5      6      7
     0   2.00   3.34   2.78   2.78   2.77   2.78   2.79   2.77
     1   3.27   2.03   2.75   2.73   2.74   2.76   2.79   2.74
     2   3.27   2.75   2.17   2.75   2.74   2.74   2.75   2.76
     3   3.27   2.74   2.76   1.99   2.77   2.74   2.76   2.79
     4   3.35   2.84   2.37   2.84   1.97   2.83   2.87   2.84
     5   3.33   2.79   2.82   2.85   2.81   2.27   2.81   2.85
     6   2.33   2.27   2.29   2.26   2.26   2.27   1.98   2.28
     7   2.81   2.83   2.85   2.84   2.82   2.82   2.78   2.03

   CPU     0      1      2      3      4      5      6      7
     0   2.29   1.93   1.86   1.86   1.85   1.85   1.89   1.87
     1   1.82   2.24   1.79   1.79   1.76   1.77   1.77   1.77
     2   2.01   1.90   2.30   1.93   1.89   1.94   1.94   1.96
     3   1.95   1.91   1.90   2.23   2.29   1.88   1.97   1.92
     4   1.83   1.80   1.86   1.80   2.06   1.79   1.80   1.86
     5   1.85   1.81   1.79   1.80   1.85   2.13   1.83   1.88
     6   1.84   1.80   1.79   1.81   1.81   1.84   2.07   1.80
     7   1.85   1.83   1.77   1.80   1.84   1.79   1.80   2.09

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled."
<img width="1270" height="156" alt="image" src="https://github.com/user-attachments/assets/51d48311-fb8d-4538-80e6-8e05add1db2f" />
