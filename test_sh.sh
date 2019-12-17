#!/bin/bash

MY_PYTHON="python"


for(( i = 5; i <= 100; i = i + 5 ))
do

  $MY_PYTHON mytest.py --i $i

done