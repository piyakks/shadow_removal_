
# shadow_removal_
<img width="3765" height="1880" alt="그림1" src="https://github.com/user-attachments/assets/69f73733-9286-4c9d-8b63-3e8b3a3f8b8f" />
heira_based_shadow_removal

## Environment
- **OS:** Ubuntu 20.04  
- **Python:** 3.10.18  
- **PyTorch:** 2.0.1  
- **CUDA:** 11.7  
- **GPU:** RTX 3090 24GB

---
## training
1. **기본 가중치 다운로드**  
[heira_l_adapter.pth]([https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/hiera) 를 다운로드 후 `sam_weight/` 폴더에 저장.

```bash
sam_weight
 └─ heira_l_adapter.pth
```
2. 학습
```bash
python train.py
```
## inference

```bash
python test.py
```
## Result
