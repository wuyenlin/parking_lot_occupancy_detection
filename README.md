# parking_lot_occupancy_detection

This repository contains the code to reproduce the result of [Deep learning for decentralized parking lot occupancy detection](https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X).
More details regarding the paper can be found on [CNRPark+EXT](http://cnrpark.it/), where dataset and labels could be downloaded.

### Run
Clone the repository and download the image dataset. Run the code as follows:

> python3 [main.py](main.py)

By default, it runs `epochs=18`, train on `CNRPark Even` and test on `CNRPark Odd`.
The setting can be changed as shown in follows. For example, 

> python3 main.py --epochs 6 --train_img PKLot/PKLotSegmented/ --train_lab splits/PKLot/UFPR04.txt --test_img PKLot/PKLotSegmented/ --test_lab splits/PKLot/UFPR04_test.txt

If a trained model is to be loaded and test on other dataset ( i.e. `.pth` file exists), or AlexNet is to be used, run the following command:

> python3 main.py --path sunny.pth --model AlexNet

See arguments in [options.py](utils/option.py).

### Requirements
```
python >= 3.6
pytorch >= 0.4
```