# DDAMGCN-pytorch-main 

## 1. Title

### Multiscale Spatio-Temporal Graph Convolutional Networks with Dynamic Delay Awareness for Traffic Forecasting

## 2. Framework
![image](Framework.png) 

### Requirements
python 3   

see requirements.txt

## 3. Train Commands
```
python train.py --force True --city ShenZhen_City --model  DDAMGCN --k_num 50
```
## 4. Test Commands
```
python test.py --force True --city ShenZhen_City --model  DDAMGCN --k_num 50
```
## 5. File directory description
eg:

```
filetree 
├── /data/ 
├── /garage/
├── README.md
├── model.py
├── test.py
├── DDAMGCN__best_model.pth
├── train.py
├── utils.py
├── util.py
├── pre_get_trend.py
├── requirements.txt
```
## Cite
## If you find this work useful, please cite it as follows.
```
@ARTICLE{11219198,
  author={Zhu, Guodong and Zhang, Xingyi and Niu, Yunyun and Du, Songzhi},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multiscale Spatiotemporal Graph Convolutional Networks With Dynamic Delay Awareness for Traffic Forecasting}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Delays;Spatiotemporal phenomena;Computational modeling;Forecasting;Correlation;Convolution;Market research;Data models;Predictive models;Noise;Dynamic delay awareness;graph convolution network (GCN);intelligent transportation system;spatiotemporal data;traffic forecasting},
  doi={10.1109/TNNLS.2025.3617860}}
```
