#!/bin/bash

python3 main.py -m picture -i 2500 -acc 0.05 -f dispersion
python3 main.py -m video -i 2500  -f red -c red 
python3 main.py -m video -i 2500  -f green -c green
python3 main.py -m video -i 2500  -f blue -c blue 
python3 main.py -m video -acc 0.05 -i 2500  -f accumulate -c all