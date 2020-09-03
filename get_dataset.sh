#!/bin/sh

which unzip > /dev/null && which wget > /dev/null && which tar > /dev/null || { echo "Please install wget, unzip, and tar" && exit 1; }

echo "This command will download and unzip the datasets used in this project."

# 1. CNRPark dataset
wget "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip" -O "CNRPark.zip"
mkdir CNRPark-Patches-150x150
unzip CNRPark.zip -d CNRPark-Patches-150x150 && rm CNRPark.zip

# 2. CNRPark-EXT dataset
wget "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "CNRPark_EXT.zip"
unzip CNRPark_EXT.zip && rm CNRPark_EXT.zip

# 3. PKLot dataset
wget "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz" -O "PKLot.tar.gz"
tar -xvf PKLot.tar.gz && rm PKLot.tar.gz

# 4. splits
wget "http://cnrpark.it/dataset/splits.zip" -O "splits.zip"
unzip splits.zip && rm splits.zip