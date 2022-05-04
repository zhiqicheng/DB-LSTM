# DB-LSTM: Densely-Connected Bi-directional LSTM for Human Action Recognition
Code for paper "[DB-LSTM: Densely-Connected Bi-directional LSTM for Human Action Recognition](https://www.sciencedirect.com/science/article/pii/S0925231220317859)"

## Requirements
* python3 
* pytorch

## Installation
```
mkdir $DYNETDIR
git clone https://github.com/zhiqicheng/DB-LSTM
```


## Example command
Training:
```
python main.py
```

Testing:
```
python eval_score.py
```

## Major changes:
- major refactoring of internal data handling

## Citation

If you find our code useful, please consider citing:
```
@article{he2021db,
  title={DB-LSTM: Densely-connected Bi-directional LSTM for human action recognition},
  author={He, Jun-Yan and Wu, Xiao and Cheng, Zhi-Qi and Yuan, Zhaoquan and Jiang, Yu-Gang},
  journal={Neurocomputing},
  volume={444},
  pages={319--331},
  year={2021},
  publisher={Elsevier}
}
```


