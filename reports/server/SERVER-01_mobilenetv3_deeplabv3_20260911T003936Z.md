# Baseline training report

run_id=SERVER-01_mobilenetv3_deeplabv3_20260911T003936Z
model=mobilenetv3_deeplabv3
config=configs/experiment/default.yaml
task_status=1

## Log tail

```text
Baseline training started at 2026-09-11T00:39:36Z (model=mobilenetv3_deeplabv3 config=configs/experiment/default.yaml)
baseline=mobilenetv3_deeplabv3 
experiment_id=baseline_mobilenetv3_deeplabv3_seed0 config_hash=329fb9468379 
device=cuda
train dataset size: 4575
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/train_baseline.py", line 46, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/train_baseline.py", line 42, in main
    run_baseline_training(args.model, config, dataloader, output_dir, console=console, device=args.device)
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/training/baseline_trainer.py", line 60, in run_baseline_training
    logits = model(image)
             ^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/models/baselines.py", line 56, in forward
    out: Tensor = self.model(image)["out"]
                  ^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torchvision/models/segmentation/_utils.py", line 23, in forward
    features = self.backbone(x)
               ^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torchvision/models/_utils.py", line 69, in forward
    x = module(x)
        ^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torchvision/models/mobilenetv3.py", line 111, in forward
    result = self.block(input)
             ^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(
           ^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 42.00 MiB. GPU 0 has a total capacity of 47.27 GiB of which 9.38 MiB is free. Process 644963 has 5.59 MiB memory in use. Process 644977 has 5.59 MiB memory in use. Process 2925282 has 312.00 MiB memory in use. Process 3053273 has 29.04 GiB memory in use. Process 3120922 has 14.74 GiB memory in use. Including non-PyTorch memory, this process has 3.14 GiB memory in use. Of the allocated memory 2.92 GiB is allocated by PyTorch, and 34.62 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```
