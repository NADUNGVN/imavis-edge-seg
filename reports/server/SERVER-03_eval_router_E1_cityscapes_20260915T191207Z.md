loaded checkpoint step=100000 config_hash=6d1274f40e43
 fit n=250 (mean error=0.1472, median=0.1308) error_risk_target=0.1472 (auto) raw_risk_target=0.4679 (auto)
    router strategies -- cityscapes -- probe=tiny     
                 target=E1/hailo_hef                  
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            3.720 │        0.3004 │
│ static_large    │           41.181 │        0.5238 │
│ entropy         │            4.757 │        0.3378 │
│ calibrated_risk │            4.966 │        0.3481 │
│ oracle          │           41.181 │        0.5238 │
└─────────────────┴──────────────────┴───────────────┘
wrote reports/router_eval_E1_cityscapes.json

task_status=0
