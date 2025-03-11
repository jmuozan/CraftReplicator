```python
python3 extract_frames.py porcelain.mp4 db/ -r 5
```


```python
python3 2_resize.py extracted_frames --max-dimension 500
```

```python
python3 3_segmenter.py
```