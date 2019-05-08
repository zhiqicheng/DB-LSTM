# ## DB-LSTM

Densely-connected Bi-directional LSTM (DB-LSTM) network

### Requirements

* python3 
* ptorch

## Installation

```
mkdir $DYNETDIR
git clone https://github.com/zhiqicheng/DB-LSTM
```


#### Example command

Training:

```
python main.py --dynet-mem 1500 --train data/da-ud-train.conllu --test data/da-ud-test.conllu --iters 10 --model da
```


Testing:
```
python main.py --model da --test data/da-ud-test.conllu --output predictions/test-da.out
```



#### Major changes:

- major refactoring of internal data handling
- renaming to `structbilty`
- `--pred-layer` is no longer required
- a single `--model` options handles both saving and loading model parameters
- the option of running a CRF has been added
- the tagger can handle additional lexical features (see our DsDs paper, EMNLP 2018) below 
- grouping of arguments
- `simplebilty` is deprecated (still available in the [former release](https://github.com/bplank/bilstm-aux/releases/tag/v1.0)




