#!/usr/bin/env python
# coding: utf-8

import scipy.optimize
import json
import numpy as np
import re
import sys
import math
import argparse
from collections import defaultdict
from pprint import pprint

def float_list(s):
    return [float(x) for x in s.split(",")] if s else []

parser = argparse.ArgumentParser()
parser.add_argument("--transport-power-cost", type=float, default=50.0,
    help="added power cost for transport per conveyor/pipeline of mined resource")
parser.add_argument("--drone-battery-cost", type=float, default=0.5,
    help="added battery cost for drone transport per conveyor/pipeline of mined resource")
parser.add_argument("--machine-penalty", type=float, default=2000.0,
    help="objective penalty per machine built")
parser.add_argument("--conveyor-penalty", type=float, default=0.0,
    help="objective penalty per conveyor belt needed")
parser.add_argument("--pipeline-penalty", type=float, default=0.0,
    help="objective penalty per pipeline needed")
parser.add_argument("--power-shard-penalty-ratio", type=float, default=0.6,
    help="objective penalty per power shard used, specified as ratio of machine penalty")
parser.add_argument("--extra-miner-clocks", type=float_list, default=[],
    help="extra clock choices for miners, specified as decimals")
parser.add_argument("--extra-manufacturer-clocks", type=float_list, default=[0.25, 0.5, 0.75],
    help="extra clock choices for manufacturers, specified as decimals")
parser.add_argument("--allow-waste", action="store_true",
    help="allow accumulation of nuclear waste and other unsinkable items")
parser.add_argument("--show-unused", action="store_true",
    help="show unused LP columns (coeff 0) in the optimization result")
parser.add_argument("--xlsx-report", type=str, default="Report.xlsx",
    help="path to xlsx report output")
parser.add_argument("--xlsx-sheet-suffix", type=str, default="",
    help="suffix to add to xlsx sheet names")
args = parser.parse_args()

### Constants ###

# Common
STACK_SIZES = {
    "SS_HUGE": 500,
    "SS_BIG": 200,
    "SS_MEDIUM": 100,
    "SS_SMALL": 50,
    "SS_ONE": 1,
    "SS_FLUID": 50000,
}
MACHINE_POWER_SHARD_LIMIT = 3
EPSILON = 1e-9

# Logistics
CONVEYOR_BELT_CLASS = "Build_ConveyorBeltMk5_C"
PIPELINE_CLASS = "Build_PipelineMK2_C"

# Resource extraction
MINER_CLASS = "Build_MinerMk3_C"
OIL_EXTRACTOR_CLASS = "Build_OilPump_C"
WATER_EXTRACTOR_CLASS = "Build_WaterPump_C"
RESOURCE_WELL_EXTRACTOR_CLASS = "Build_FrackingExtractor_C"
RESOURCE_WELL_PRESSURIZER_CLASS = "Build_FrackingSmasher_C"

# Sink
SINK_CLASS = "Build_ResourceSink_C"

# Water
WATER_CLASS = "Desc_Water_C"

# Nuclear power
NUCLEAR_WASTE_MAPPINGS = {
    "Desc_NuclearFuelRod_C": "Desc_NuclearWaste_C",
    "Desc_PlutoniumFuelRod_C": "Desc_PlutoniumWaste_C",
}

# Geothermal power
GEOTHERMAL_GENERATOR_CLASS = "Build_GeneratorGeoThermal_C"
GEYSER_CLASS = "Desc_Geyser_C"

# Resource map
PURITY_MULTIPLIERS = {
    "impure": 0.5,
    "normal": 1.0,
    "pure": 2.0,
}
POWER_SLUG_SHARDS = {
    "greenSlugs": 1,
    "yellowSlugs": 2,
    "purpleSlugs": 5,
}
RESOURCE_MAPPINGS = {
    "Desc_LiquidOilWell_C": "Desc_LiquidOil_C",
    "Desc_SAM_C": None,  # exclude
}

# Miscellaneous
BIOMASS_GENERATOR_CLASS = "Build_GeneratorBiomass_C"
BATTERY_CLASS = "Desc_Battery_C"
ADDITIONAL_ITEMS = {
    "Desc_PlutoniumWaste_C": {
        "class": "Desc_PlutoniumWaste_C",
        "display_name": "Plutonium Waste",
        "form": "RF_SOLID",
        "points": 0,
        "stack_size": STACK_SIZES["SS_HUGE"],
        "energy": 0.0,
    },
}
ADDITIONAL_DISPLAY_NAMES = {
    GEYSER_CLASS: "Geyser",
}


docs_path = r"Docs.json"
map_info_path = r"MapInfo.json"

with open(docs_path, "r", encoding="utf-16") as f:
    docs_raw = json.load(f)

class_entries = {}
class_types = {}

for fg_entry in docs_raw:
    class_type = re.sub(r"Class'/Script/FactoryGame.(\w+)'", r"\1", fg_entry["NativeClass"])
    class_type_list = []
    for class_entry in fg_entry["Classes"]:
        class_name = class_entry["ClassName"]
        if class_name in class_entries:
            print(f"WARNING: ignoring duplicate class {class_name}")
        else:
            class_entries[class_name] = class_entry
            class_type_list.append(class_entry)
    class_types[class_type] = class_type_list


### Parsing helpers ###

def parse_paren_list(s):
    if not s:
        return None
    assert(s.startswith("(") and s.endswith(")"))
    s = s[1:-1]
    if not s:
        return []
    else:
        return s.split(",")

def find_class_name(s):
    m = re.search(r"\.\w+", s)
    if m is None:
        raise ValueError(f"could not find class name in: {s}")
    return m[0][1:]

def parse_class_list(s):
    l = parse_paren_list(s)
    if l is None:
        return l
    return [find_class_name(x) for x in l]

def find_item_amounts(s):
    for m in re.finditer(r"\(ItemClass=([^,]+),Amount=(\d+)\)", s):
        yield (find_class_name(m[1]), int(m[2]))


### Misc constants ###

CONVEYOR_BELT_LIMIT = 0.5 * float(class_entries[CONVEYOR_BELT_CLASS]["mSpeed"])
PIPELINE_LIMIT = 60000.0 * float(class_entries[PIPELINE_CLASS]["mFlowLimit"])
SINK_POWER_CONSUMPTION = float(class_entries[SINK_CLASS]["mPowerConsumption"])

print(f"CONVEYOR_BELT_LIMIT: {CONVEYOR_BELT_LIMIT}")
print(f"PIPELINE_LIMIT: {PIPELINE_LIMIT}")
print(f"SINK_POWER_CONSUMPTION: {SINK_POWER_CONSUMPTION}")


### Miners ###

def parse_miner(entry):
    if entry["ClassName"] == RESOURCE_WELL_PRESSURIZER_CLASS:
        extractor = class_entries[RESOURCE_WELL_EXTRACTOR_CLASS]
    else:
        extractor = entry

    return {
        "class": entry["ClassName"],
        "display_name": entry["mDisplayName"],
        "power_consumption": float(entry["mPowerConsumption"]),
        "power_consumption_exponent": float(entry["mPowerConsumptionExponent"]),
        "min_clock": float(entry["mMinPotential"]),
        "max_clock_base": float(entry["mMaxPotential"]),
        "max_clock_per_power_shard": float(entry["mMaxPotentialIncreasePerCrystal"]),
        "rate": 60.0 / float(extractor["mExtractCycleTime"]) * float(extractor["mItemsPerCycle"]),
        "only_allow_certain_resources": (extractor["mOnlyAllowCertainResources"] == "True"),
        "allowed_resource_forms": parse_paren_list(extractor["mAllowedResourceForms"]),
        "allowed_resources": parse_class_list(extractor["mAllowedResources"]),
    }

miners = {}
for name in (MINER_CLASS, OIL_EXTRACTOR_CLASS, WATER_EXTRACTOR_CLASS, RESOURCE_WELL_PRESSURIZER_CLASS):
    miners[name] = parse_miner(class_entries[name])

# pprint(miners)


### Manufacturers ###

def parse_manufacturer(entry):
    return {
        "class": entry["ClassName"],
        "display_name": entry["mDisplayName"],
        "power_consumption": float(entry["mPowerConsumption"]),
        "power_consumption_exponent": float(entry["mPowerConsumptionExponent"]),
        "min_clock": float(entry["mMinPotential"]),
        "max_clock_base": float(entry["mMaxPotential"]),
        "max_clock_per_power_shard": float(entry["mMaxPotentialIncreasePerCrystal"]),
    }

manufacturers = {}

for entry in class_types["FGBuildableManufacturer"]:
    manufacturer = parse_manufacturer(entry)
    manufacturer["is_variable_power"] = False
    manufacturers[entry["ClassName"]] = manufacturer

for entry in class_types["FGBuildableManufacturerVariablePower"]:
    manufacturer = parse_manufacturer(entry)
    manufacturer["is_variable_power"] = True
    manufacturers[entry["ClassName"]] = manufacturer

# pprint(manufacturers)


### Recipes ###

def parse_recipe(entry):
    recipe_manufacturer = None
    for manufacturer in parse_class_list(entry["mProducedIn"]) or []:
        if manufacturer in manufacturers:
            recipe_manufacturer = manufacturer
            break

    # we are only considering automatable recipes
    if recipe_manufacturer is None:
        return None

    rate = 60.0 / float(entry["mManufactoringDuration"])
    def item_rates(key):
        return [(item, rate * amount) for (item, amount) in find_item_amounts(entry[key])]

    vpc_constant = float(entry["mVariablePowerConsumptionConstant"])
    vpc_factor = float(entry["mVariablePowerConsumptionFactor"])

    return {
        "class": entry["ClassName"],
        "display_name": entry["mDisplayName"],
        "manufacturer": recipe_manufacturer,
        "inputs": item_rates("mIngredients"),
        "outputs": item_rates("mProduct"),
        "variable_power_consumption": vpc_constant + 0.5 * vpc_factor,
    }

recipes = {}
for entry in class_types["FGRecipe"]:
    recipe = parse_recipe(entry)
    if recipe is not None:
        recipes[entry["ClassName"]] = recipe

# pprint(recipes)


### Items ###

def parse_item(entry):
    points = int(entry["mResourceSinkPoints"])
    return {
        "display_name": entry["mDisplayName"],
        "form": entry["mForm"],
        "points": int(entry["mResourceSinkPoints"]),
        "stack_size": STACK_SIZES[entry["mStackSize"]],
        "energy": float(entry["mEnergyValue"]),
    }

items = {}

# any items not contained in Docs.json
items.update(ADDITIONAL_ITEMS)

for class_type in [
    "FGItemDescriptor",
    "FGItemDescriptorBiomass",
    "FGItemDescriptorNuclearFuel",
    "FGResourceDescriptor",
    "FGEquipmentDescriptor",
    "FGConsumableDescriptor",
]:
    for entry in class_types[class_type]:
        item = parse_item(entry)
        if class_type == "FGItemDescriptorNuclearFuel":
            item["nuclear_waste"] = NUCLEAR_WASTE_MAPPINGS[entry["ClassName"]]
            item["nuclear_waste_amount"] = float(entry["mAmountOfWaste"])
        items[entry["ClassName"]] = item

# pprint(items)


### Generators ###

generators = {}

def parse_generator(entry):
    power_production = float(entry["mPowerProduction"])
    return {
        "display_name": entry["mDisplayName"],
        "fuel_classes": parse_class_list(entry["mDefaultFuelClasses"]),
        "power_production": power_production,
        "power_production_exponent": float(entry["mPowerProductionExponent"]),
        "requires_supplemental": (entry["mRequiresSupplementalResource"] == "True"),
        "supplemental_to_power_ratio": float(entry["mSupplementalToPowerRatio"]),
    }

def parse_geothermal_generator(entry):
    # unclear why mVariablePowerProductionConstant=0 in the json;
    # it's set to 100.0f in the header, which we will hardcode here
    return {
        "display_name": entry["mDisplayName"],
        "power_production": 100.0 + 0.5 * float(entry["mVariablePowerProductionFactor"]),
    }

# coal and fuel generators
for entry in class_types["FGBuildableGeneratorFuel"]:
    # exclude biomass generator
    if entry["ClassName"] == BIOMASS_GENERATOR_CLASS:
        continue

    generators[entry["ClassName"]] = parse_generator(entry)

# nuclear power plant
for entry in class_types["FGBuildableGeneratorNuclear"]:
    generators[entry["ClassName"]] = parse_generator(entry)

# geothermal generator (special case)
geothermal_generator = parse_geothermal_generator(class_entries[GEOTHERMAL_GENERATOR_CLASS])

# pprint(generators)


### Resources ###

with open(map_info_path, "r") as f:
    map_info_raw = json.load(f)

map_info = {}

for tab in map_info_raw["options"]:
    if "tabId" in tab:
        map_info[tab["tabId"]] = tab["options"]

TOTAL_POWER_SHARDS = 0
for slug_type in map_info["power_slugs"][0]["options"]:
    TOTAL_POWER_SHARDS += POWER_SLUG_SHARDS[slug_type["layerId"]] * len(slug_type["markers"])

print(f"TOTAL_POWER_SHARDS: {TOTAL_POWER_SHARDS}")

resources = {}
geysers = {}

def parse_and_add_node_type(node_type):
    if "type" not in node_type:
        return

    item = node_type["type"]
    if item in RESOURCE_MAPPINGS:
        item = RESOURCE_MAPPINGS[item]

    if item is None:
        return

    output = geysers if item == GEYSER_CLASS else resources

    for node_purity in node_type["options"]:
        purity = node_purity["purity"]
        nodes = node_purity["markers"]
        if not nodes:
            continue
        sample_node = nodes[0]
        if "core" in sample_node:
            # resource well satellite nodes, map them to cores
            for node in nodes:
                subtype = find_class_name(node["core"])
                resource_id = f"{item}|{subtype}"
                if resource_id not in output:
                    output[resource_id] = {
                        "resource_id": resource_id,
                        "item": item,
                        "subtype": subtype,
                        "multiplier": 0.0,
                        "count": 1,
                        "is_limited": True,
                        "is_resource_well": True,
                        "num_satellites": 0,
                    }
                output[resource_id]["multiplier"] += PURITY_MULTIPLIERS[purity]
                output[resource_id]["num_satellites"] += 1
        else:
            # normal nodes, add directly
            subtype = purity
            resource_id = f"{item}|{subtype}"
            assert(resource_id not in output)
            output[resource_id] = {
                "resource_id": resource_id,
                "item": item,
                "subtype": subtype,
                "multiplier": PURITY_MULTIPLIERS[purity],
                "count": len(nodes),
                "is_limited": True,
                "is_resource_well": False,
            }

for node_type in map_info["resource_nodes"]:
    parse_and_add_node_type(node_type)

for node_type in map_info["resource_wells"]:
    parse_and_add_node_type(node_type)

resources[WATER_CLASS] = {
    "resource_id": f"{WATER_CLASS}:extractor",
    "item": WATER_CLASS,
    "subtype": "extractor",
    "multiplier": 1,
    "is_limited": False,
    "is_resource_well": False,
}

# pprint(resources)
# pprint(geysers)


### LP setup ###

class LPColumn(dict):
    def __init__(self, *args, display_info=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_info = display_info

lp_columns = {}
lp_equalities = {}
lp_lower_bounds = {}

def get_power_consumption(machine, clock=1.0, recipe=None):
    power_consumption = machine["power_consumption"]
    if recipe is not None and machine.get("is_variable_power", False):
        power_consumption += recipe["variable_power_consumption"]

    return power_consumption * (clock ** machine["power_consumption_exponent"])

def get_miner_for_resource(resource):
    item_class = resource["item"]
    item = items[item_class]
    candidates = []
    for miner_class, miner in miners.items():
        if ((resource["is_resource_well"]) == (miner_class == RESOURCE_WELL_PRESSURIZER_CLASS)
            and item["form"] in miner["allowed_resource_forms"]
            and (not miner["only_allow_certain_resources"] or item_class in miner["allowed_resources"])):
            candidates.append(miner_class)
    if not candidates:
        raise RuntimeError(f"could not find miner for resource {item_class}")
    elif len(candidates) > 1:
        raise RuntimeError(f"more than one miner for resource {item_class}: {candidates}")
    return candidates[0]

def get_form_conveyance_limit(form):
    if form == "RF_SOLID":
        return CONVEYOR_BELT_LIMIT
    elif form == "RF_LIQUID" or form == "RF_GAS":
        return PIPELINE_LIMIT
    else:
        assert(False)

def get_max_overclock(machine):
    return machine["max_clock_base"] + MACHINE_POWER_SHARD_LIMIT * machine["max_clock_per_power_shard"]

def get_conveyance_limit_clock(item, rate):
    conveyance_limit = get_form_conveyance_limit(item["form"])
    return math.floor(1000000 * conveyance_limit / rate) / 1000000

def get_max_miner_clock(miner, resource, rate):
    max_overclock = get_max_overclock(miner)

    if resource["is_resource_well"]:
        return max_overclock

    item_class = resource["item"]
    item = items[item_class]
    return min(max_overclock, get_conveyance_limit_clock(item, rate))

def get_max_manufacturer_clock(manufacturer, recipe):
    max_clock = get_max_overclock(manufacturer)

    for (item_class, rate) in recipe["inputs"]:
        max_clock = min(max_clock, get_conveyance_limit_clock(items[item_class], rate))

    for (item_class, rate) in recipe["outputs"]:
        max_clock = min(max_clock, get_conveyance_limit_clock(items[item_class], rate))

    return max_clock

def get_power_shards_needed(machine, clock):
    return max(0, math.ceil((clock - machine["max_clock_base"]) / machine["max_clock_per_power_shard"]))

def get_item_display_name(item_class):
    if item_class in items:
        return items[item_class]["display_name"]
    else:
        return ADDITIONAL_DISPLAY_NAMES[item_class]

def add_lp_column(column, type_, name, display_name=None, machine_name=None, subtype=None, clock=None):
    tokens = [type_, name]
    if subtype is not None:
        tokens.append(subtype)
    if clock is not None:
        clock_percent = 100.0 * clock
        tokens.append(f"{clock_percent}")
    column_id = "|".join(tokens)
    display_info = {
        "type": type_,
        "display_name": display_name or name,
        "machine_name": machine_name,
        "subtype": subtype,
        "clock": clock,
    }
    lp_columns[column_id] = LPColumn(column, display_info=display_info)

for resource_id, resource in resources.items():
    item_class = resource["item"]
    item = items[item_class]

    miner_class = get_miner_for_resource(resource)
    miner = miners[miner_class]

    rate = miner["rate"] * resource["multiplier"]
    min_clock = miner["min_clock"]
    max_clock_base = miner["max_clock_base"]
    max_clock = get_max_miner_clock(miner, resource, rate)

    resource_var = f"resource|{resource_id}"
    item_var = f"item|{item_class}"

    clock_choices = {max_clock_base, max_clock}
    for clock in args.extra_miner_clocks:
        clock = min(max_clock, max(min_clock, clock))
        clock_choices.add(clock)

    for clock in sorted(clock_choices):
        column = {
            item_var: clock * rate,
            "power_consumption": get_power_consumption(miner, clock=clock),
            "machines": 1 + (resource["num_satellites"] if resource["is_resource_well"] else 0),
        }

        if resource["is_limited"]:
            column[resource_var] = -1

        power_shards = get_power_shards_needed(miner, clock)
        if power_shards > 0:
            column["power_shard_usage"] = power_shards

        add_lp_column(
            column,
            type_="miner",
            name=resource_id,
            display_name=item["display_name"],
            machine_name=miner["display_name"],
            subtype=resource["subtype"],
            clock=clock,
        )

    if resource["is_limited"]:
        lp_lower_bounds[resource_var] = -resource["count"]

    lp_equalities[item_var] = 0.0

for recipe_class, recipe in recipes.items():
    manufacturer_class = recipe["manufacturer"]
    manufacturer = manufacturers[manufacturer_class]

    min_clock = manufacturer["min_clock"]
    max_clock_base = manufacturer["max_clock_base"]
    max_clock = get_max_manufacturer_clock(manufacturer, recipe)

    # let's not allow manufacturer OC by default, but it can be specified via option
    clock_choices = {min_clock, max_clock_base}
    for clock in args.extra_manufacturer_clocks:
        clock = min(max_clock, max(min_clock, clock))
        clock_choices.add(clock)

    for clock in sorted(clock_choices):
        column = {
            "power_consumption": get_power_consumption(manufacturer, clock=clock, recipe=recipe),
            "machines": 1,
        }

        for (item_class, rate) in recipe["inputs"]:
            item_var = f"item|{item_class}"
            column[item_var] = column.get(item_var, 0.0) - clock * rate
            lp_equalities[item_var] = 0.0

        for (item_class, rate) in recipe["outputs"]:
            item_var = f"item|{item_class}"
            column[item_var] = column.get(item_var, 0.0) + clock * rate
            lp_equalities[item_var] = 0.0

        power_shards = get_power_shards_needed(manufacturer, clock)
        if power_shards > 0:
            column["power_shard_usage"] = power_shards

        add_lp_column(
            column,
            type_="manufacturer",
            name=recipe_class,
            display_name=recipe["display_name"],
            machine_name=manufacturer["display_name"],
            clock=clock,
        )

for item_class, item in items.items():
    points = item["points"]
    item_var = f"item|{item_class}"

    if not (item["form"] == "RF_SOLID" and points > 0):
        if args.allow_waste:
            add_lp_column(
                {item_var: -1},
                type_="waste",
                name=item_class,
                display_name=item["display_name"],
            )
        continue

    column = {
        item_var: -1,
        "points": points,
        "power_consumption": SINK_POWER_CONSUMPTION / CONVEYOR_BELT_LIMIT,
        "machines": 1 / CONVEYOR_BELT_LIMIT,
    }

    add_lp_column(
        column,
        type_="sink",
        name=item_class,
        display_name=item["display_name"],
    )

    lp_equalities[item_var] = 0.0

for generator_class, generator in generators.items():
    power_production = generator["power_production"]
    for fuel_class in generator["fuel_classes"]:
        fuel = items[fuel_class]
        fuel_rate = 60.0 * power_production / fuel["energy"]
        fuel_var = f"item|{fuel_class}"

        column = {
            fuel_var: -fuel_rate,
            "power_production": power_production,
            "machines": 1,
        }

        if generator["requires_supplemental"]:
            supplemental_class = WATER_CLASS
            supplemental_var = f"item|{supplemental_class}"
            supplemental_rate = 60.0 * power_production * generator["supplemental_to_power_ratio"]
            column[supplemental_var] = -supplemental_rate
            lp_equalities[supplemental_var] = 0.0

        if fuel_class in NUCLEAR_WASTE_MAPPINGS:
            waste_class = NUCLEAR_WASTE_MAPPINGS[fuel_class]
            waste_var = f"item|{waste_class}"
            column[waste_var] = fuel_rate * fuel["nuclear_waste_amount"]
            lp_equalities[waste_var] = 0.0

        add_lp_column(
            column,
            type_="generator",
            name=fuel_class,
            display_name=fuel["display_name"],
            machine_name=generator["display_name"],
            clock=1,
        )

for resource_id, resource in geysers.items():
    resource_var = f"resource|{resource_id}"

    column = {
        resource_var: -1,
        "power_production": geothermal_generator["power_production"] * resource["multiplier"],
        "machines": 1,
    }

    add_lp_column(
        column,
        type_="generator",
        name=resource_id,
        display_name=get_item_display_name(GEYSER_CLASS),
        machine_name=geothermal_generator["display_name"],
        subtype=resource["subtype"],
    )

    lp_lower_bounds[resource_var] = -resource["count"]

for column_id, column in lp_columns.items():
    to_add = defaultdict(float)
    for variable, coeff in column.items():
        if abs(coeff) < EPSILON:
            print(f"WARNING: zero or near-zero coeff: column_id={column_id} variable={variable} coeff={coeff}")

        if variable.startswith("item|") and coeff > 0:
            item_class = variable[5:]
            if item_class not in items:
                print(f"WARNING: item not found in items dict: {item_class}")
                continue

            item = items[item_class]
            form = item["form"]
            conveyance_limit = get_form_conveyance_limit(form)
            conveyance = coeff / conveyance_limit

            if column_id.startswith("miner|"):
                to_add["transport_power_cost"] += args.transport_power_cost * conveyance
                to_add["drone_battery_cost"] += args.drone_battery_cost * conveyance

            if form == "RF_SOLID":
                to_add["conveyors"] += conveyance
            else:
                to_add["pipelines"] += conveyance

    for variable, coeff in to_add.items():
        if coeff != 0.0:
            column[variable] = column.get(variable, 0.0) + coeff

for objective in ["points", "machines", "conveyors", "pipelines"]:
    column = {
        objective: -1,
    }
    add_lp_column(
        column,
        type_="objective",
        name=objective,
    )
    lp_equalities[objective] = 0.0

for extra_cost, cost_variable, cost_coeff in [
    ("transport_power_cost", "power_consumption", 1.0),
    ("drone_battery_cost", f"item|{BATTERY_CLASS}", -1.0),
]:
    column = {
        extra_cost: -1,
        cost_variable: cost_coeff,
    }
    add_lp_column(
        column,
        type_="extra_cost",
        name=extra_cost,
    )
    lp_equalities[extra_cost] = 0.0

column = {
    "power_consumption": -1,
    "power_production": -1,
}
add_lp_column(
    column,
    type_="power",
    name="usage",
)
lp_equalities["power_consumption"] = 0.0
lp_lower_bounds["power_production"] = 0.0

column = {
    "power_shard_usage": -1,
    "power_shards": -1,
}
add_lp_column(
    column,
    type_="objective",
    name="power_shards",
)
lp_equalities["power_shard_usage"] = 0.0
lp_lower_bounds["power_shards"] = -TOTAL_POWER_SHARDS

# pprint(lp_columns)
# pprint(lp_equalities)
# pprint(lp_lower_bounds)


def get_all_variables():
    variables = set()

    for column_id, column in lp_columns.items():
        for variable, coeff in column.items():
            variables.add(variable)

    for variable in variables:
        if variable not in lp_equalities and variable not in lp_lower_bounds:
            print(f"WARNING: no constraint for variable: {variable}")

    for variable in lp_equalities.keys():
        if variable not in variables:
            print(f"WARNING: equality constraint with unknown variable: {variable}")

    for variable in lp_lower_bounds.keys():
        if variable not in variables:
            print(f"WARNING: lower bound constraint with unknown variable: {variable}")

    return variables

variables = get_all_variables()
# pprint(variables)


### Pruning ###

reachable_items = set()
while True:
    any_added = False
    for column_id, column in lp_columns.items():
        eligible = True
        to_add = set()
        for variable, coeff in column.items():
            if variable.startswith("item|") and variable not in reachable_items:
                if coeff > 0:
                    to_add.add(variable)
                elif coeff < 0:
                    eligible = False
                    break
        if eligible and to_add:
            any_added = True
            reachable_items |= to_add
    if not any_added:
        break

unreachable_items = set(v for v in variables if v.startswith("item|")) - reachable_items

print("pruning unreachable items:")
pprint(unreachable_items)

columns_to_prune = list()
for column_id, column in lp_columns.items():
    for variable, coeff in column.items():
        if variable in unreachable_items and coeff < 0:
            columns_to_prune.append(column_id)
            break
for column_id in columns_to_prune:
    # pprint(lp_columns[column_id])
    del lp_columns[column_id]
for item_var in unreachable_items:
    if item_var in lp_equalities:
        del lp_equalities[item_var]

variables = get_all_variables()
# pprint(variables)

# pprint(lp_columns)
# pprint(lp_equalities)
# pprint(lp_lower_bounds)


### LP run ###

def to_index_map(seq):
    return {value: index for index, value in enumerate(seq)}

def from_index_map(d):
    result = [None] * len(d)
    for value, index in d.items():
        result[index] = value
    return result

# order is for report display, but we might as well sort it here
column_type_order = to_index_map(["objective", "power", "extra_cost", "sink", "waste", "manufacturer", "miner", "generator"])
column_subtype_order = to_index_map(["impure", "normal", "pure"])
objective_order = to_index_map(["points", "machines", "conveyors", "pipelines", "power_shards"])
extra_cost_order = to_index_map(["transport_power_cost", "drone_battery_cost"])

def column_order_key(arg):
    column_id, column = arg
    info = column.display_info

    type_ = info["type"]
    if type_ in column_type_order:
        type_key = (0, column_type_order[type_])
    else:
        type_key = (1, type_)

    name = info["display_name"]
    if type_ == "objective":
        name_key = objective_order[name]
    elif type_ == "extra_cost":
        name_key = extra_cost_order[name]
    else:
        name_key = name

    subtype = info["subtype"]
    if subtype in column_subtype_order:
        subtype_key = (0, column_subtype_order[subtype])
    else:
        subtype_key = (1, subtype)

    return (type_key, name_key, subtype_key, info["clock"], column_id)

sorted_columns = sorted(lp_columns.items(), key=column_order_key)
indices_eq = to_index_map(sorted(lp_equalities.keys()))
indices_lb = to_index_map(sorted(lp_lower_bounds.keys()))

# pprint(indices_eq)
# pprint(indices_lb)

lp_c = np.zeros(len(lp_columns), dtype=np.double)
lp_A_eq = np.zeros((len(lp_equalities), len(lp_columns)), dtype=np.double)
lp_b_eq = np.zeros(len(lp_equalities), dtype=np.double)
lp_A_lb = np.zeros((len(lp_lower_bounds), len(lp_columns)), dtype=np.double)
lp_b_lb = np.zeros(len(lp_lower_bounds), dtype=np.double)

objective_weights = {f"objective|{obj}": weight for (obj, weight) in {
    "points": 1,
    "machines": -args.machine_penalty,
    "conveyors": -args.conveyor_penalty,
    "pipelines": -args.pipeline_penalty,
    "power_shards": -args.power_shard_penalty_ratio * args.machine_penalty,
}.items()}

for column_index, (column_id, column) in enumerate(sorted_columns):
    if column_id in objective_weights:
        lp_c[column_index] = objective_weights[column_id]
    for variable, coeff in column.items():
        if variable in lp_equalities:
            lp_A_eq[indices_eq[variable], column_index] = coeff
        else:
            lp_A_lb[indices_lb[variable], column_index] = coeff

for variable, rhs in lp_equalities.items():
    lp_b_eq[indices_eq[variable]] = rhs

for variable, rhs in lp_lower_bounds.items():
    lp_b_lb[indices_lb[variable]] = rhs

print("running LP")

lp_result = scipy.optimize.linprog(-lp_c, A_ub=-lp_A_lb, b_ub=-lp_b_lb, A_eq=lp_A_eq, b_eq=lp_b_eq, method="highs")

if lp_result.status != 0:
    print("ERROR: LP did not terminate successfully")
    pprint(lp_result)
    sys.exit(1)

pprint(lp_result)


### Display formatting ###

def format_subtype(subtype):
    if subtype is None or subtype == "extractor":
        return None
    return re.sub(r"^BP_FrackingCore_?", "#", subtype).capitalize()

def get_column_desc(column):
    info = column.display_info
    tokens = [info["machine_name"] or info["type"], info["display_name"]]
    subtype = format_subtype(info["subtype"])
    if subtype is not None:
        tokens.append(subtype)
    if info["clock"] is not None:
        clock_percent = 100.0 * info["clock"]
        tokens.append(f"{clock_percent}%")
    return "|".join(tokens)

column_results = [
    (column_id, column, lp_result.x[column_index])
    for column_index, (column_id, column) in enumerate(sorted_columns)
]

if not args.show_unused:
    column_results = list(filter(lambda x: abs(x[2]) > EPSILON, column_results))

variable_breakdowns = {variable: {"production": [], "consumption": []} for variable in variables}

for column_id, column, column_coeff in column_results:
    column_desc = get_column_desc(column)

    print(f"{column_desc} = {column_coeff}")

    for variable, coeff in column.items():
        rate = column_coeff * coeff
        source = {
            "desc": column_desc,
            "count": column_coeff,
            "rate": abs(rate),
        }
        if abs(rate) < EPSILON:
            continue
        elif rate > 0:
            variable_breakdowns[variable]["production"].append(source)
        else:
            variable_breakdowns[variable]["consumption"].append(source)

variable_order = to_index_map(
    from_index_map(objective_order)
    + ["power_production", "power_consumption"]
    + from_index_map(extra_cost_order)
    + ["power_shards", "power_shard_usage", "item", "resource"]
)

def get_variable_display_info(variable):
    tokens = variable.split("|")
    type_ = tokens[0]
    if type_ == "item" or type_ == "resource":
        item_class = tokens[1]
        tokens[1] = get_item_display_name(item_class)
    return (type_, "|".join(tokens))

def finalize_variable_budget_side(budget_side):
    if not budget_side:
        return
    total_rate = 0.0
    for entry in budget_side:
        total_rate += entry["rate"]
    for entry in budget_side:
        entry["share"] = entry["rate"] / total_rate
    budget_side.sort(key=lambda entry: (-entry["share"], entry["desc"]))
    budget_side.insert(0, {"desc": "Total", "count": "n/a", "rate": total_rate, "share": 1.0})

for variable, breakdown in variable_breakdowns.items():
    type_, name = get_variable_display_info(variable)

    # don't show offsetting dummy items in the breakdown (e.g. "objective|points" as consumer of points)
    # currently these are precisely the consumption of special variables, but that may change
    if type_ not in ["item", "resource"]:
        breakdown["consumption"] = []

    breakdown["type_order"] = variable_order[type_]
    breakdown["name"] = name
    if variable in indices_lb:
        slack = lp_result.slack[indices_lb[variable]]
        if slack < -EPSILON:
            print(f"WARNING: lower bound violation: variable={variable} slack={slack}")
        breakdown["initial"] = -lp_lower_bounds[variable]
        breakdown["final"] = slack
    else:
        con = lp_result.con[indices_eq[variable]]
        if abs(con) > EPSILON:
            print(f"WARNING: equality constraint violation: variable={variable} con={con}")
    finalize_variable_budget_side(breakdown["production"])
    finalize_variable_budget_side(breakdown["consumption"])

sorted_variable_breakdowns = sorted(variable_breakdowns.values(), key=lambda bd: (bd["type_order"], bd["name"]))

# pprint(sorted_variable_breakdowns)

if args.xlsx_report:
    print("writing xlsx report")

    import xlsxwriter
    workbook = xlsxwriter.Workbook(args.xlsx_report)

    default_format = workbook.add_format({"align": "center"})
    top_format = workbook.add_format({"align": "center", "top": True})
    bold_format = workbook.add_format({"align": "center", "bold": True})
    bold_underline_format = workbook.add_format({"align": "center", "bold": True, "underline": True})
    bold_top_format = workbook.add_format({"align": "center", "bold": True, "top": True})
    bold_underline_top_format = workbook.add_format({"align": "center", "bold": True, "underline": True, "top": True})
    percent_format = workbook.add_format({"align": "center", "num_format": "0.0#####%"})

    sheet1 = workbook.add_worksheet("List" + args.xlsx_sheet_suffix)
    sheet2 = workbook.add_worksheet("Breakdown" + args.xlsx_sheet_suffix)

    def write_cell(sheet, *args, fmt=default_format):
        sheet.write(*args, fmt)

    sheet1.add_table(0, 0, len(column_results), 5, {
        "columns": [{"header": header, "header_format": bold_format}
        for header in ["Type", "Name", "Machine", "Subtype", "Clock", "Quantity"]],
        "style": "Table Style Light 16",
    })

    for i, (column_id, column, column_coeff) in enumerate(column_results):
        info = column.display_info
        write_cell(sheet1, i + 1, 0, info["type"])
        write_cell(sheet1, i + 1, 1, info["display_name"])
        write_cell(sheet1, i + 1, 2, info["machine_name"] or "n/a")
        write_cell(sheet1, i + 1, 3, info["subtype"] or "n/a")
        write_cell(sheet1, i + 1, 4, info["clock"] or "n/a", fmt=percent_format)
        write_cell(sheet1, i + 1, 5, column_coeff)

    for c, width in enumerate([14, 39, 25, 19, 11, 13]):
        sheet1.set_column(c, c, width)

    current_row = 0
    max_budget_entries = 0
    budget_rows = [
        ("desc", "Producer", "Consumer"),
        ("count", "Producer Count", "Consumer Count"),
        ("rate", "Production Rate", "Consumption Rate"),
        ("share", "Production Share", "Consumption Share"),
    ]

    production_share_cf = {
        "type": "2_color_scale",
        "min_type": "num",
        "max_type": "num",
        "min_value": 0,
        "max_value": 1,
        "min_color": "#FFFFFF",
        "max_color": "#99FF99"
    }
    consumption_share_cf = production_share_cf.copy()
    consumption_share_cf["max_color"] = "#FFCC66"

    for variable_index, breakdown in enumerate(sorted_variable_breakdowns):
        for budget_side_index, budget_side_name in enumerate(["production", "consumption"]):
            budget_side = breakdown[budget_side_name]
            if not budget_side:
                continue
            for budget_row in budget_rows:
                key = budget_row[0]
                name = budget_row[budget_side_index + 1]
                if key == "desc":
                    fmts = (bold_top_format, bold_underline_top_format)
                elif key == "share":
                    fmts = (bold_format, percent_format)
                else:
                    fmts = (bold_format, default_format)
                write_cell(sheet2, current_row, 0, breakdown["name"], fmt=fmts[0])
                write_cell(sheet2, current_row, 1, name, fmt=fmts[0])
                for i, entry in enumerate(budget_side):
                    write_cell(sheet2, current_row, i + 2, entry[key], fmt=fmts[1])
                if key == "share":
                    cf = production_share_cf if budget_side_name == "production" else consumption_share_cf
                    sheet2.conditional_format(current_row, 3, current_row, len(budget_side) + 1, cf)
                max_budget_entries = max(max_budget_entries, len(budget_side))
                current_row += 1
        for key in ["initial", "final"]:
            if key in breakdown:
                if key == "initial":
                    fmts = (bold_top_format, top_format)
                else:
                    fmts = (bold_format, default_format)
                fmt = bold_top_format if key == "initial" else bold_format
                write_cell(sheet2, current_row, 0, breakdown["name"], fmt=fmts[0])
                write_cell(sheet2, current_row, 1, key.capitalize(), fmt=fmts[0])
                write_cell(sheet2, current_row, 2, breakdown[key], fmt=fmts[1])
                current_row += 1

    for c, width in enumerate([41, 19, 13] + [59] * (max_budget_entries - 1)):
        sheet2.set_column(c, c, width)

    workbook.close()
