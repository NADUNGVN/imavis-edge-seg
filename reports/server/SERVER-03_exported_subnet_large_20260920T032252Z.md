# Exported-subnet training report

run_id=SERVER-03_exported_subnet_large_20260920T032252Z
level=large
fp32_checkpoint=outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt
config=configs/experiment/default.yaml
task_status=1

## Log tail

```text
Exported-subnet training started at 2026-09-20T03:22:52Z (level=large fp32_checkpoint=outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt config=configs/experiment/default.yaml)
exported_subnet level=large experiment_id=qat_exported_large_dynamic_seed0 
config_hash=f23a16e937ea device=cuda qat=True calibrate=False
extracted large subnet from 
outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt (step=100000)
train dataset size: 4575
 QAT enabled: all nn.Conv2d layers fake-quantized (INT8)
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/train_exported_subnet.py", line 129, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/train_exported_subnet.py", line 110, in main
    run_baseline_training(
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/training/baseline_trainer.py", line 157, in run_baseline_training
    loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap, not ours
    ^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 23.56 GiB of which 1.40 GiB is free. Process 123548 has 4.85 GiB memory in use. Including non-PyTorch memory, this process has 17.29 GiB memory in use. Of the allocated memory 15.78 GiB is allocated by PyTorch, and 1.21 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```
