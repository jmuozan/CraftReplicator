```python
python3 1_extract_frames.py celadon_vessel.mp4 extracted_frames -r 5
```


```python
python3 2_resize.py extracted_frames --max-dimension 1000
```

```python
python3 3_segmenter.py
```







```python
python 5_split_rename.py --input extracted_objects --output celadon_vessel --test 15
```

