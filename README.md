# SatisfactoryLP

Satisfactory linear programming optimization, using `scipy.optimize.milp`.

Recipes are parsed from data provided by Coffee Stain Studios in the Satisfactory game directory.

Map data for resource node availability is taken from [Satisfactory Calculator](https://satisfactory-calculator.com/). I don't know how to obtain it directly.

Data sources, retrieved 2026-07-17:
- `Docs.json`: `Satisfactory\CommunityResources\Docs\en-US.json` (v1.2.3.1 - Build 495413)
- `MapInfo.json`: `https://static.satisfactory-calculator.com/data/json/mapData/en-Stable.json?v=1784184789` (plus auto-formatting; lastBuild 492064)

[Baseline results on Google Sheets](https://docs.google.com/spreadsheets/d/1Vkklgd37jbtgURB6zjLq7--5rMnRe5FfQ9EoWpQuh40/edit?usp=sharing), generated 2024-11-04.

**Update 1.2:** Recipes and resources unchanged since 1.0. In the data, the base power consumption of the variable power machines (Particle Accelerator, Quantum Encoder, Converter) was reduced from 0.1 MW to 0 MW. It's unclear if this reflects an actual in-game change.

# Requirements
- Python 3.12 (older versions may work)
- `pip install scipy xlsxwriter`

# Running
Running `python SatisfactoryLP.py` produces some text output and a `Results.xlsx` file which can be opened in Excel or imported to Google Sheets. There are various options to control extra costs and penalty weights; use `-h` to see details.

The most important options are:
- `--machine-penalty` controls the tradeoff between underclocking and machine count. Increasing this penalty causes the optimizer to prefer higher clock speeds and fewer machines. On the other hand, decreasing it to zero would cause most machines to be set to the lowest possible clock speed.
- `--machine-limit` sets a hard limit for the number of machines, which is another way to control this tradeoff. If using `--machine-limit`, you usually also want `--machine-penalty=0` to remove any soft limit. I have saved some results at various machine limits in `Frontier.csv`. These were obtained using a 1% clock speed granularity, which is finer than the default and takes a while to run.
- `--manufacturer-clocks` specifies the allowed clock speeds for non-somerslooped manufacturers and Water Extractors. Increasing the granularity allows the optimizer to reach higher objectives, at the cost of increased problem size and running time. All clock speed options can be specified as a comma-separated list, where each token is either (1) a single clock speed `VALUE` (2) a triple `LO-HI/STEP` (linear), or (3) a triple `LO-HI/xRATIO` (geometric). The default setting is `0-2.5/0.05` which means "every clock speed between 0% and 250% at 5% increments." (Note that 0% will automatically be forced up to the minumum clock speed of 1%, and higher clock speeds will automatically be forced down if they would exceed a conveyor belt limit for the recipe.)
- `--num-somersloops-available` allows you to override the total number of Somersloops available for production amplifiers and Alien Power Augmenters, after research costs. By default, this number is 105, since the map has 106 Somersloops, but 1 Somersloop is needed to unlock Production Amplification in the MAM, and it is not optimal to unlock Somersloop Analysis or Alien Power Augmenters in the MAM. For a standard save where the MAM was fully researched, set `--num-somersloops-available=103`. For an AGS save with No Unlock Cost or Unlock All Research in the MAM, set `--num-somersloops-available=106`.
- `--power-consumption-multiplier` is a setting from Update 1.2. At 5x power consumption, adding one fueled Alien Power Augmenter becomes optimal, and fine-tuning underclocks becomes more significant. I recommend the settings: `--power-consumption-multiplier=5 --num-alien-power-augmenters=1 --num-fueled-alien-power-augmenters=1 --manufacturer-clocks=0.01-2.5/x1.05 --miner-clocks=1.5-2.5/x1.05`.

TODO: document more options.
