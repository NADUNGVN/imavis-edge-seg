loaded checkpoint step=100000 config_hash=6d1274f40e43
 fit n=250 (mean error=0.1472, median=0.1308) error_risk_target=0.0736 (explicit) raw_risk_target=0.2340 (explicit)
    router strategies -- cityscapes -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.3004 │
│ static_large    │            9.414 │        0.5238 │
│ entropy         │            2.893 │        0.4195 │
│ calibrated_risk │            3.172 │        0.4251 │
│ oracle          │            9.414 │        0.5238 │
└─────────────────┴──────────────────┴───────────────┘
wrote reports/router_eval_E3_cityscapes_strict.json

task_status=0
