# parking_lot_occupancy_detection

This repository contains the code to reproduce the result of [Deep learning for decentralized parking lot occupancy detection](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X).
More details regarding the paper can be found on [CNRPark+EXT](http://cnrpark.it/), where dataset and labels could be downloaded.

### Running the code
Clone the repository and download the image dataset. Run the code as follows:

> python3 [main.py](main.py)

By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`. 
If a trained model is to be loaded and test on other dataset (i.e. `.pth` file exists), or AlexNet is to be used, run the following command:

> python3 main.py --path trained_model/sunny.pth --model AlexNet

See arguments in [options.py](utils/option.py).

### Requirements
```
python >= 3.6
pytorch >= 0.4
```

### Results
Results of Table 2 are shown below, with epochs=18.

|Test set | Paper | Pytorch |
|-----	  |-----  | -----   |
|Trained on UFPR04	    |
|UFPR04   | 0.9954| 0.9590  |
|UFPR05   | 0.9329| 0.7520  |
|PUC	  | 0.9827| 0.9040  |
|Trained on UFPR05	    |
|UFPR04   | 0.9369| 0.8460  |
|UFPR05   | 0.9949| 0.9740  |
|PUC	  | 0.9272| 0.9070  |
|Trained on PUC		    |
|UFPR04   | 0.9803| 0.9440  |
|UFPR05   | 0.9600| 0.9330  |
|PUC	  | 0.9990| 0.9880  |

### References
Amato, Giuseppe and Carrara, Fabio and Falchi, Fabrizio and Gennaro, Claudio and Meghini, Carlo and Vairo, Claudio. Deep learning for decentralized parking lot occupancy detection. Expert Systems with Applications (Pergamon), 2017.
