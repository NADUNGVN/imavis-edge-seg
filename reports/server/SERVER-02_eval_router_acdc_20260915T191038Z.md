loaded checkpoint step=100000 config_hash=6d1274f40e43
 fit n=50 (mean error=0.1302, median=0.1279) error_risk_target=0.1302 (auto) raw_risk_target=0.4295 (auto)
     router strategies -- acdc/fog -- probe=tiny      
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.2981 │
│ static_large    │            9.414 │        0.5104 │
│ entropy         │            1.540 │        0.3602 │
│ calibrated_risk │            1.676 │        0.3658 │
│ oracle          │            9.414 │        0.5104 │
└─────────────────┴──────────────────┴───────────────┘
 fit n=53 (mean error=0.3003, median=0.2649) error_risk_target=0.3003 (auto) raw_risk_target=0.6964 (auto)
    router strategies -- acdc/night -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.1824 │
│ static_large    │            9.414 │        0.3554 │
│ entropy         │            1.506 │        0.2254 │
│ calibrated_risk │            1.554 │        0.2282 │
│ oracle          │            9.414 │        0.3554 │
└─────────────────┴──────────────────┴───────────────┘
 fit n=50 (mean error=0.1276, median=0.1346) error_risk_target=0.1276 (auto) raw_risk_target=0.4766 (auto)
     router strategies -- acdc/rain -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.2760 │
│ static_large    │            9.414 │        0.4874 │
│ entropy         │            1.403 │        0.3240 │
│ calibrated_risk │            1.591 │        0.3423 │
│ oracle          │            9.414 │        0.4874 │
└─────────────────┴──────────────────┴───────────────┘
 fit n=50 (mean error=0.1879, median=0.1695) error_risk_target=0.1879 (auto) raw_risk_target=0.5178 (auto)
     router strategies -- acdc/snow -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.2477 │
│ static_large    │            9.414 │        0.4735 │
│ entropy         │            1.420 │        0.3270 │
│ calibrated_risk │            1.454 │        0.3283 │
│ oracle          │            9.414 │        0.4735 │
└─────────────────┴──────────────────┴───────────────┘
wrote reports/router_eval_E3_acdc.json

task_status=0
