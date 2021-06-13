# SatisfactoryLP
Satisfactory linear programming optimization, using `scipy.optimize.linprog`. Most data is parsed from the `Docs.json` provided by Coffee Stain Studios in the Satisfactory game directory. Also uses `MapInfo.json` taken from [Satisfactory Calculator](https://satisfactory-calculator.com/) for resource node information.

[Baseline results on Google Sheets](https://docs.google.com/spreadsheets/d/1q4dvdzhLXdbDdV1lxKM27TFOuI2TariiiItFIuygWEw/edit?usp=sharing) (might not be regularly updated).

# Requirements
- Python 3.8+
- `pip install scipy xlsxwriter`

# Running
Running `python SatisfactoryLP.py` produces some text output and a `Results.xlsx` file which can be opened in Excel or imported to Google Sheets. There are various options to control extra costs and penalty weights; use `-h` to see details.
