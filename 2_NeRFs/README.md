```bash
python3 1_split_rename.py --input "imgs" --output "porcelain/imgs" --test 10
```



```bash
python3 colmap2nerf.py --run_colmap --images porcelain/imgs --colmap_matcher exhaustive --colmap_camera_model SIMPLE_PINHOLE
```

```bash

```