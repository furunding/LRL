#!/bin/bash

for i in $(seq $2 $3)  
do  
    echo docker rm -f $1_$i;  
done