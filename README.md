# SatisfactoryLP

Satisfactory linear programming optimization, using `scipy.optimize.milp`.

Recipes are parsed from data provided by Coffee Stain Studios in the Satisfactory game directory.

Map data for resource node availability is taken from [Satisfactory Calculator](https://satisfactory-calculator.com/). I don't know how to obtain it directly.

Data sources, retrieved 2024-11-04:
- `Docs.json`: `Satisfactory\CommunityResources\Docs\en-US.json` (v1.0.0.5 - Build 377620)
- `MapInfo.json`: `https://static.satisfactory-calculator.com/data/json/mapData/en-Stable.json?v=1730198931` (plus auto-formatting)

[Baseline results on Google Sheets](https://docs.google.com/spreadsheets/d/1Vkklgd37jbtgURB6zjLq7--5rMnRe5FfQ9EoWpQuh40/edit?usp=sharing), generated 2024-11-04.

# Requirements
- Python 3.12 (older versions may work)
- `pip install scipy xlsxwriter`

# Running
Running `python SatisfactoryLP.py` produces some text output and a `Results.xlsx` file which can be opened in Excel or imported to Google Sheets. There are various options to control extra costs and penalty weights; use `-h` to see details.

The most important options are:
- `--machine-penalty` controls the tradeoff between underclocking and machine count. Increasing this penalty causes the optimizer to prefer higher clock speeds and fewer machines. On the other hand, decreasing it to zero would cause most machines to be set to the lowest possible clock speed.
- `--machine-limit` sets a hard limit for the number of machines, which is another way to control this tradeoff. If using `--machine-limit`, you usually also want `--machine-penalty=0` to remove any soft limit. I have saved some results at various machine limits in `Frontier.csv`. These were obtained using a 1% clock speed granularity, which is finer than the default and takes a while to run.
- `--manufacturer-clocks` specifies the allowed clock speeds for non-somerslooped manufacturers and Water Extractors. Increasing the granularity allows the optimizer to reach higher objectives, at the cost of increased problem size and running time. All clock speed options can be specified as a comma-separated list, where each token is either a single clock speed or a triple `lower-upper/step`. The default setting is `0-2.5/0.05` which means "every clock speed between 0% and 250% at 5% increments." This is somewhat overkill and results in 51 versions of each recipe being created; however, it still runs in under 3 seconds on my machine. For faster runs, a reasonable setting is `0-1/0.25,1.5-2.5/0.5` which allows clock speeds [0%, 25%, 50%, 75%, 100%, 150%, 200%, 250%]. This runs in about 0.7 seconds and the resulting objective is only 0.12% worse. (Note that 0% will automatically be forced up to the minumum clock speed of 1%, and higher clock speeds will automatically be forced down if they would exceed a conveyor belt limit for the recipe.)
- `--num-somersloops-available` allows you to override the total number of Somersloops available for production amplifiers and Alien Power Augmenters, after research costs. By default, this number is 105, since the map has 106 Somersloops, but 1 Somersloop is needed to unlock Production Amplification in the MAM, and it is not optimal to unlock Somersloop Analysis or Alien Power Augmenters in the MAM. For a standard save where the MAM was fully researched, set `--num-somersloops-available=103`. For an AGS save with No Unlock Cost or Unlock All Research in the MAM, set `--num-somersloops-available=106`.

TODO: document more options.
