## This is a Demo

##### You should prepare Data First
```sh
cd  data
chmod a+x download_deep_lesion.sh
./download_deep_lesion.sh
```
When Download finished
##### run the matlab script prepare_deep_lesion.m

##### train the model
```sh
python train.py
```
the result will be in runs

##### test the model
```sh
python test.py
```
the result willbe in directory testt
