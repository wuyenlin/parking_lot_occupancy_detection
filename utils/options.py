import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=18, help="rounds of training")
    parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
    parser.add_argument('--model', type=str, default='mAlexNet', help='model name')
    parser.add_argument('--path', type=str, default='', help='trained model path')
    parser.add_argument('--train_img', type=str, default='CNRPark-Patches-150x150/', help="path to training set images")
    parser.add_argument('--train_lab', type=str, default='splits/CNRParkAB/even.txt', help="path to training set labels")
    parser.add_argument('--test_img', type=str, default='CNRPark-Patches-150x150/', help="path to test set images")
    parser.add_argument('--test_lab', type=str, default='splits/CNRParkAB/odd.txt', help="path to test set labels")
    args = parser.parse_args()
    return args