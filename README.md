# Parking Lot Occupancy Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zau9rpEkNuihJca9BI6raiRaiNOwj_-j?usp=sharing) 

This repository contains the code to reproduce the result of [Deep learning for decentralized parking lot occupancy detection](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X).
More details regarding the paper can be found on [CNRPark+EXT](http://cnrpark.it/), where dataset and labels could be downloaded. 
This reproduction code is done by Hao Liu, Sigurd Totland, and Yen-Lin Wu.

### Download dataset
There are 3 sets of dataset and their labels required to run this code. Run [get_dataset.sh](get_dataset.sh) as follows:

```
sudo chmod +x get_dataset.sh
./get_dataset.sh
```

This command will download the datasets and unzip them in the project root directory.
Make sure to include the correct directory in the next section when parsing the argument.


### Running the code
Run the code as follows:

```
python3 main.py
```

By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`. 
If a trained model is to be loaded and test on other dataset (i.e. `.pth` file exists), or AlexNet is to be used, run the following command:

```
python3 main.py --path trained_model/sunny.pth --model AlexNet
```

See arguments in [options.py](utils/option.py).

### Requirements
```
python >= 3.6
pytorch >= 0.4
```

### Results
For the moment, only Table 2 and Figure 5 are reproduced from the paper. Some variances could be observed from the results compared to paper. The optimal epochs for each experiment are still being worked on.

Results of Table 2 are shown below, with epochs=18.

|Test set | Paper | Pytorch |
|-----	  |-----  | -----   |
|Trained on UFPR04	    |
|UFPR04   | 0.9954| 0.9600  |
|UFPR05   | 0.9329| 0.7990  |
|PUC	  | 0.9827| 0.9300  |
|Trained on UFPR05	    |
|UFPR04   | 0.9369| 0.8000  |
|UFPR05   | 0.9949| 0.9760  |
|PUC	  | 0.9272| 0.9010  |
|Trained on PUC		    |
|UFPR04   | 0.9803| 0.9560  |
|UFPR05   | 0.9600| 0.9490  |
|PUC	  | 0.9990| 0.9890  |
|Trained on CNRParkOdd
|CNRParkEven|0.9013|0.9190  |
|Trained on CNRParkEven
|CNRParkOdd|0.9071| 0.9240  |


Results of Figure 5 are shown below.

Paper results:

|Test set | Paper | Pytorch |
|-----	  |-----  | -----   |
|Trained on SUNNY	    |
|OVERCAST | 0.970  | 0.946 |
|RAINY    | 0.960  | 0.912 |
|PKLot    | 0.850  | 0.759 |
|Trained on OVERCAST      |
|SUNNY    | 0.920 | 0.917 |
|RAINY    | 0.950 | 0.920 |
|PKLot    | 0.820 | 0.709 |
|Trained on RAINY	  |
|SUNNY    | 0.940 | 0.914 |
|OVERCAST | 0.970 | 0.959 |
|PKLot    | 0.920 | 0.651 |


### References
Amato, Giuseppe and Carrara, Fabio and Falchi, Fabrizio and Gennaro, Claudio and Meghini, Carlo and Vairo, Claudio. Deep learning for decentralized parking lot occupancy detection. Expert Systems with Applications (Pergamon), 2017.
