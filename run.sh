#!/bin/bash

echo "Compile and run..."

BUILDING_DIR="build"
if [ -d "$BUILDING_DIR" ]; then rm -Rf $BUILDING_DIR; fi

mkdir build
cd build 
cmake -DDEAL_II_DIR=/home/hello/Documents/dealii/install .. 
make 
make run