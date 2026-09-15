loaded checkpoint step=100000 config_hash=6d1274f40e43
 fit n=250 (mean error=0.1472, median=0.1308) error_risk_target=0.1472 raw_risk_target=0.4680 (auto = fit-half mean of the matching scale)
    router strategies -- cityscapes -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.3003 │
│ static_large    │            9.414 │        0.5238 │
│ entropy         │            1.246 │        0.3378 │
│ calibrated_risk │            1.307 │        0.3481 │
│ oracle          │            9.414 │        0.5238 │
└─────────────────┴──────────────────┴───────────────┘
wrote reports/router_eval_E3_cityscapes_fixed.json

task_status=0
