#!/usr/bin/env bash

python SatisfactoryLP.py --transport-power-cost=0 --drone-battery-cost=0 --machine-penalty=0 --xlsx-report="Report-MP0.xlsx" --xlsx-sheet-suffix=" (MP0)"

for MP in 1000 2000 4000 8000; do
    python SatisfactoryLP.py --machine-penalty=${MP} --xlsx-report="Report-MP${MP}.xlsx" --xlsx-sheet-suffix=" (MP${MP})"
done
