git_sha=395d5b10509ff0303bc84252f80bd5a4e00f1061
checkpoint step=100000 sha256=a24441568a4c
Traceback (most recent call last):
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/onnx/_internal/onnx_proto_utils.py", line 215, in _add_onnxscript_fn
    import onnx
ModuleNotFoundError: No module named 'onnx'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/prepare_compiled_eval.py", line 161, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/prepare_compiled_eval.py", line 104, in main
    path = export_subnet_onnx(sub, out / "onnx" / f"pace_seg_{level}.onnx", h, w)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/export.py", line 28, in export_subnet_onnx
    torch.onnx.export(
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/onnx/__init__.py", line 375, in export
    export(
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/onnx/utils.py", line 502, in export
    _export(
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/onnx/utils.py", line 1640, in _export
    proto = onnx_proto_utils._add_onnxscript_fn(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/onnx/_internal/onnx_proto_utils.py", line 217, in _add_onnxscript_fn
    raise errors.OnnxExporterError("Module onnx is not installed!") from e
torch.onnx.OnnxExporterError: Module onnx is not installed!

task_status=1
