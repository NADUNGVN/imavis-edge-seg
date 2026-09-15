loaded checkpoint step=100000 config_hash=6d1274f40e43
 fit n=250 (mean error=0.1472, median=0.1308) risk_target=0.1472 (auto = fit-half mean error)
    router strategies -- cityscapes -- probe=tiny     
                target=E3/tensorrt_gpu                
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ strategy        ┃ avg latency (ms) ┃ achieved mIoU ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ static_small    │            0.941 │        0.3003 │
│ static_large    │            9.414 │        0.5238 │
│ entropy         │            6.611 │        0.5032 │
│ calibrated_risk │            1.307 │        0.3481 │
│ oracle          │            9.414 │        0.5238 │
└─────────────────┴──────────────────┴───────────────┘
wrote reports/router_eval_E3_cityscapes.json

task_status=0
