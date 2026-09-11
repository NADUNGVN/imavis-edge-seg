loaded checkpoint step=100000 config_hash=9828ff7136b4 git_commit=a1c1a2af1e6e5906d67ca08f9cddf9dfbebcc1a1
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/evaluate_supernet.py", line 139, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/scripts/evaluate_supernet.py", line 126, in main
    result = evaluate_level(supernet, level, loader, device=args.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/evaluation/evaluator.py", line 29, in evaluate_level
    for image, mask in dataloader:
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 757, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/data/acdc.py", line 63, in __getitem__
    return self.transform(image, mask)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/imavis-edge-seg/src/imavis_edge_seg/data/transforms.py", line 31, in __call__
    image = image.convert("RGB").resize(
            ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/PIL/Image.py", line 1069, in convert
    self.load()
  File "/home/ubuntu/miniconda3/envs/imavis-edge-seg/lib/python3.11/site-packages/PIL/ImageFile.py", line 412, in load
    n, err_code = decoder.decode(b)
                  ^^^^^^^^^^^^^^^^^
KeyboardInterrupt
