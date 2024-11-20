#!/usr/bin/env python
# pyright: basic

from dataclasses import dataclass
from fractions import Fraction
import scipy.optimize
import json
import numpy as np
import re
import sys
import argparse
from collections import defaultdict
from pprint import pprint
from typing import Any, Iterable, TypeVar, cast


T = TypeVar("T")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--machine-penalty",
    type=float,
    default=1000.0,
    help="objective penalty per machine built",
)
parser.add_argument(
    "--conveyor-penalty",
    type=float,
    default=0.0,
    help="objective penalty per conveyor belt of machine input/output",
)
parser.add_argument(
    "--pipeline-penalty",
    type=float,
    default=0.0,
    help="objective penalty per pipeline of machine input/output",
)
parser.add_argument(
    "--machine-limit",
    type=float,
    help="hard limit on number of machines built",
)
parser.add_argument(
    "--transport-power-cost",
    type=float,
    default=0.0,
    help="added power cost to simulate transport per conveyor/pipeline of mined resource",
)
parser.add_argument(
    "--drone-battery-cost",
    type=float,
    default=0.0,
    help="added battery cost to simulate drone transport per conveyor/pipeline of mined resource",
)
parser.add_argument(
    "--miner-clocks",
    type=str,
    default="2.5",
    help="clock choices for miners (excluding Water Extractors)",
)
parser.add_argument(
    "--manufacturer-clocks",
    type=str,
    default="0-2.5/0.05",
    help="clock choices for non-somerslooped manufacturers (plus Water Extractors)",
)
parser.add_argument(
    "--somersloop-clocks",
    type=str,
    default="2.5",
    help="clock choices for somerslooped manufacturers",
)
parser.add_argument(
    "--generator-clocks",
    type=str,
    default="2.5",
    help="clock choices for power generators",
)
parser.add_argument(
    "--num-alien-power-augmenters",
    type=int,
    default=0,
    help="number of Alien Power Augmenters to build",
)
parser.add_argument(
    "--num-fueled-alien-power-augmenters",
    type=float,
    default=0,
    help="number of Alien Power Augmenters to fuel with Alien Power Matrix",
)
parser.add_argument(
    "--disable-production-amplification",
    action="store_true",
    help="disable usage of somersloops in manufacturers",
)
parser.add_argument(
    "--resource-multipliers",
    type=str,
    default="",
    help="comma-separated list of item_class:multiplier to scale resource node availability",
)
parser.add_argument(
    "--num-somersloops-available",
    type=int,
    help="override number of somersloops available for production and APAs",
)
parser.add_argument(
    "--disabled-recipes",
    type=str,
    default="",
    help="comma-separated list of recipe_class to disable",
)
parser.add_argument(
    "--infinite-power",
    action="store_true",
    help="allow free infinite power consumption",
)
parser.add_argument(
    "--allow-waste",
    action="store_true",
    help="allow accumulation of nuclear waste and other unsinkable items",
)
parser.add_argument(
    "--show-unused",
    action="store_true",
    help="show unused LP columns (coeff 0) in the optimization result",
)
parser.add_argument(
    "--dump-debug-info",
    action="store_true",
    help="dump debug info to DebugInfo.txt (items, recipes, LP matrix, etc.)",
)
parser.add_argument(
    "--xlsx-report",
    type=str,
    default="Report.xlsx",
    help="path to xlsx report output (empty string to disable)",
)
parser.add_argument(
    "--xlsx-sheet-suffix",
    type=str,
    default="",
    help="suffix to add to xlsx sheet names",
)
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

# Clock speeds
INVERSE_CLOCK_GRANULARITY = 100 * 10000


def float_to_clock(value: float) -> Fraction:
    return Fraction(round(value * INVERSE_CLOCK_GRANULARITY), INVERSE_CLOCK_GRANULARITY)


def str_to_clock(s: str) -> Fraction:
    return float_to_clock(float(s))


def clock_to_percent_str(clock: Fraction) -> str:
    return f"{float(100 * clock)}%"


MACHINE_BASE_CLOCK = float_to_clock(1.0)
MACHINE_MAX_CLOCK = float_to_clock(2.5)

# Logistics
CONVEYOR_BELT_CLASS = "Build_ConveyorBeltMk6_C"
PIPELINE_CLASS = "Build_PipelineMK2_C"

# Miners
MINER_CLASS = "Build_MinerMk3_C"
OIL_EXTRACTOR_CLASS = "Build_OilPump_C"
WATER_EXTRACTOR_CLASS = "Build_WaterPump_C"
RESOURCE_WELL_EXTRACTOR_CLASS = "Build_FrackingExtractor_C"
RESOURCE_WELL_PRESSURIZER_CLASS = "Build_FrackingSmasher_C"
ALL_MINER_CLASSES = (
    MINER_CLASS,
    OIL_EXTRACTOR_CLASS,
    WATER_EXTRACTOR_CLASS,
    RESOURCE_WELL_PRESSURIZER_CLASS,
)

# Sink
SINK_CLASS = "Build_ResourceSink_C"

# Items
ALL_ITEM_NATIVE_CLASSES = (
    "FGItemDescriptor",
    "FGItemDescriptorBiomass",
    "FGItemDescriptorNuclearFuel",
    "FGItemDescriptorPowerBoosterFuel",
    "FGResourceDescriptor",
    "FGEquipmentDescriptor",
    "FGConsumableDescriptor",
    "FGPowerShardDescriptor",
    "FGAmmoTypeProjectile",
    "FGAmmoTypeInstantHit",
    "FGAmmoTypeSpreadshot",
)

# Water
WATER_CLASS = "Desc_Water_C"

# Generators (excl. geothermal)
ALL_GENERATOR_NATIVE_CLASSES = (
    "FGBuildableGeneratorFuel",
    "FGBuildableGeneratorNuclear",
)

# Geothermal generator
GEOTHERMAL_GENERATOR_CLASS = "Build_GeneratorGeoThermal_C"
GEYSER_CLASS = "Desc_Geyser_C"

# Alien Power Augmenter
ALIEN_POWER_AUGMENTER_CLASS = "Build_AlienPowerBuilding_C"
ALIEN_POWER_MATRIX_CLASS = "Desc_AlienPowerFuel_C"

# Resource map
PURITY_MULTIPLIERS = {
    "impure": 0.5,
    "normal": 1.0,
    "pure": 2.0,
}
RESOURCE_MAPPINGS = {
    "Desc_LiquidOilWell_C": "Desc_LiquidOil_C",
}

# Miscellaneous
BATTERY_CLASS = "Desc_Battery_C"
ADDITIONAL_DISPLAY_NAMES = {
    GEYSER_CLASS: "Geyser",
}


### Debug ###


DEBUG_INFO_PATH = r"DebugInfo.txt"
PPRINT_WIDTH = 120

debug_file = (
    open(DEBUG_INFO_PATH, mode="w", encoding="utf-8") if args.dump_debug_info else None
)


def debug_dump(heading: str, obj: object):
    if debug_file is None:
        return
    print(f"========== {heading} ==========", file=debug_file)
    print("", file=debug_file)
    if isinstance(obj, str):
        print(obj, file=debug_file)
    else:
        pprint(obj, stream=debug_file, width=PPRINT_WIDTH, sort_dicts=False)
    print("", file=debug_file)


### Configured clock speeds ###


def parse_clock_spec(s: str) -> list[Fraction]:
    result: list[Fraction] = []
    for token in s.split(","):
        token = token.strip()
        if "/" in token:
            bounds, _, step_str = token.rpartition("/")
            lower_str, _, upper_str = bounds.rpartition("-")
            lower = str_to_clock(lower_str)
            upper = str_to_clock(upper_str)
            step = str_to_clock(step_str)
            current = lower
            while current <= upper:
                result.append(current)
                current += step
        else:
            result.append(str_to_clock(token))
    result.sort()
    return result


MINER_CLOCKS = parse_clock_spec(args.miner_clocks)
MANUFACTURER_CLOCKS = parse_clock_spec(args.manufacturer_clocks)
SOMERSLOOP_CLOCKS = parse_clock_spec(args.somersloop_clocks)
GENERATOR_CLOCKS = parse_clock_spec(args.generator_clocks)

debug_dump(
    "Configured clock speeds",
    f"""
{MINER_CLOCKS=}
{MANUFACTURER_CLOCKS=}
{SOMERSLOOP_CLOCKS=}
{GENERATOR_CLOCKS=}
""".strip(),
)


### Configured resource multipliers ###


def parse_resource_multipliers(s: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if s:
        for token in s.split(","):
            item_class, _, multiplier = token.partition(":")
            assert item_class not in result
            result[item_class] = float(multiplier)
    return result


RESOURCE_MULTIPLIERS = parse_resource_multipliers(args.resource_multipliers)

debug_dump(
    "Configured resource multipliers",
    f"""
{RESOURCE_MULTIPLIERS=}
""".strip(),
)


### Configured disabled recipes ###

DISABLED_RECIPES: list[str] = [
    token.strip() for token in args.disabled_recipes.split(",")
]

debug_dump(
    "Configured disabled recipes",
    f"""
{DISABLED_RECIPES=}
""".strip(),
)


### Load json ###


DOCS_PATH = r"Docs.json"
MAP_INFO_PATH = r"MapInfo.json"


with open(DOCS_PATH, "r", encoding="utf-16") as f:
    docs_raw = json.load(f)


### Initial parsing ###


class_name_to_entry: dict[str, dict[str, Any]] = {}
native_class_to_class_entries: dict[str, list[dict[str, Any]]] = {}

NATIVE_CLASS_REGEX = re.compile(r"/Script/CoreUObject.Class'/Script/FactoryGame.(\w+)'")


def parse_and_add_fg_entry(fg_entry: dict[str, Any]):
    m = NATIVE_CLASS_REGEX.fullmatch(fg_entry["NativeClass"])
    assert m is not None, fg_entry["NativeClass"]
    native_class = m.group(1)

    class_entries: list[dict[str, Any]] = []
    for class_entry in fg_entry["Classes"]:
        class_name = class_entry["ClassName"]
        if class_name in class_name_to_entry:
            print(f"WARNING: ignoring duplicate class {class_name}")
        else:
            class_name_to_entry[class_name] = class_entry
            class_entries.append(class_entry)
    native_class_to_class_entries[native_class] = class_entries


for fg_entry in docs_raw:
    parse_and_add_fg_entry(fg_entry)


### Parsing helpers ###


def parse_paren_list(s: str) -> list[str] | None:
    if not s:
        return None
    assert s.startswith("(") and s.endswith(")")
    s = s[1:-1]
    if not s:
        return []
    else:
        return s.split(",")


QUALIFIED_CLASS_NAME_REGEX = re.compile(r"\"?/Script/[^']+'/[\w\-/]+\.(\w+)'\"?")
UNQUALIIFIED_CLASS_NAME_REGEX = re.compile(r"\"?/[\w\-/]+\.(\w+)\"?")


def extract_class_name(s: str) -> str:
    m = QUALIFIED_CLASS_NAME_REGEX.fullmatch(
        s
    ) or UNQUALIIFIED_CLASS_NAME_REGEX.fullmatch(s)
    assert m is not None, s
    return m.group(1)


def parse_class_list(s: str) -> list[str] | None:
    l = parse_paren_list(s)
    if l is None:
        return None
    return [extract_class_name(x) for x in l]


ITEM_AMOUNT_REGEX = re.compile(r"\(ItemClass=([^,]+),Amount=(\d+)\)")


def find_item_amounts(s: str) -> Iterable[tuple[str, int]]:
    for m in ITEM_AMOUNT_REGEX.finditer(s):
        yield (extract_class_name(m[1]), int(m[2]))


### Misc constants ###


CONVEYOR_BELT_LIMIT = 0.5 * float(class_name_to_entry[CONVEYOR_BELT_CLASS]["mSpeed"])
PIPELINE_LIMIT = 60000.0 * float(class_name_to_entry[PIPELINE_CLASS]["mFlowLimit"])
SINK_POWER_CONSUMPTION = float(class_name_to_entry[SINK_CLASS]["mPowerConsumption"])

debug_dump(
    "Misc constants",
    f"""
{CONVEYOR_BELT_LIMIT=}
{PIPELINE_LIMIT=}
{SINK_POWER_CONSUMPTION=}
""".strip(),
)


ALIEN_POWER_AUGMENTER_STATIC_POWER = float(
    class_name_to_entry[ALIEN_POWER_AUGMENTER_CLASS]["mBasePowerProduction"]
)
ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST = float(
    class_name_to_entry[ALIEN_POWER_AUGMENTER_CLASS]["mBaseBoostPercentage"]
)
ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST = (
    ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST
    + float(class_name_to_entry[ALIEN_POWER_MATRIX_CLASS]["mBoostPercentage"])
)
ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE = 60.0 / float(
    class_name_to_entry[ALIEN_POWER_MATRIX_CLASS]["mBoostDuration"]
)


debug_dump(
    "Alien Power Augmenter constants",
    f"""
{ALIEN_POWER_AUGMENTER_STATIC_POWER=}
{ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST=}
{ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST=}
{ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE=}
""".strip(),
)


### Classes ###


@dataclass
class ClassObject:
    class_name: str
    display_name: str


@dataclass
class Machine(ClassObject):
    min_clock: Fraction
    max_clock: Fraction


@dataclass
class PowerConsumer(Machine):
    power_consumption: float
    power_consumption_exponent: float
    is_variable_power: bool


@dataclass
class Miner(PowerConsumer):
    extraction_rate_base: float
    uses_resource_wells: bool
    allowed_resource_forms: list[str]
    only_allow_certain_resources: bool
    allowed_resources: list[str] | None

    def check_allowed_resource(self, item_class: str, form: str) -> bool:
        if form not in self.allowed_resource_forms:
            return False
        if self.only_allow_certain_resources:
            assert self.allowed_resources
            return item_class in self.allowed_resources
        else:
            return True


@dataclass
class Manufacturer(PowerConsumer):
    can_change_production_boost: bool
    base_production_boost: float
    production_shard_slot_size: int
    production_shard_boost_multiplier: float
    production_boost_power_consumption_exponent: float


@dataclass
class Recipe(ClassObject):
    manufacturer: str
    inputs: list[tuple[str, float]]
    outputs: list[tuple[str, float]]
    mean_variable_power_consumption: float


@dataclass
class Item(ClassObject):
    class_name: str
    display_name: str
    form: str
    points: int
    stack_size: int
    energy: float


@dataclass
class Fuel:
    fuel_class: str
    supplemental_resource_class: str | None
    byproduct: str | None
    byproduct_amount: int


@dataclass
class PowerGenerator(Machine):
    fuels: list[Fuel]
    power_production: float
    requires_supplemental: bool
    supplemental_to_power_ratio: float


@dataclass
class GeothermalGenerator(Machine):
    mean_variable_power_production: float


### Miners ###


def parse_miner(entry: dict[str, Any]) -> Miner:
    # Resource well extractors are aggregated under the pressurizer
    if entry["ClassName"] == RESOURCE_WELL_PRESSURIZER_CLASS:
        extractor = class_name_to_entry[RESOURCE_WELL_EXTRACTOR_CLASS]
        uses_resource_wells = True
    else:
        extractor = entry
        uses_resource_wells = False

    assert entry["mCanChangePotential"] == "True"
    assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

    assert entry["mCanChangeProductionBoost"] == "False"

    # This will be multiplied by the purity when applied to a node
    # (or, for resource wells, the sum of satellite purities).
    extraction_rate_base = (
        60.0
        / float(extractor["mExtractCycleTime"])
        * float(extractor["mItemsPerCycle"])
    )

    allowed_resource_forms = parse_paren_list(extractor["mAllowedResourceForms"])
    assert allowed_resource_forms is not None

    return Miner(
        class_name=entry["ClassName"],
        display_name=entry["mDisplayName"],
        power_consumption=float(entry["mPowerConsumption"]),
        power_consumption_exponent=float(entry["mPowerConsumptionExponent"]),
        min_clock=str_to_clock(entry["mMinPotential"]),
        max_clock=MACHINE_MAX_CLOCK,
        is_variable_power=False,
        extraction_rate_base=extraction_rate_base,
        uses_resource_wells=uses_resource_wells,
        allowed_resource_forms=allowed_resource_forms,
        only_allow_certain_resources=(
            extractor["mOnlyAllowCertainResources"] == "True"
        ),
        allowed_resources=parse_class_list(extractor["mAllowedResources"]),
    )


miners: dict[str, Miner] = {}

for class_name in ALL_MINER_CLASSES:
    miners[class_name] = parse_miner(class_name_to_entry[class_name])

debug_dump("Parsed miners", miners)


### Manufacturers ###


def parse_manufacturer(entry: dict[str, Any], is_variable_power: bool) -> Manufacturer:
    class_name = entry["ClassName"]

    assert entry["mCanChangePotential"] == "True"
    assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

    can_change_production_boost = entry["mCanChangeProductionBoost"] == "True"
    production_shard_slot_size = int(entry["mProductionShardSlotSize"])

    # Smelter has "mProductionShardSlotSize": "0" when it should be 1
    if class_name == "Build_SmelterMk1_C":
        assert can_change_production_boost
        if production_shard_slot_size == 0:
            production_shard_slot_size = 1

    return Manufacturer(
        class_name=class_name,
        display_name=entry["mDisplayName"],
        power_consumption=float(entry["mPowerConsumption"]),
        power_consumption_exponent=float(entry["mPowerConsumptionExponent"]),
        min_clock=str_to_clock(entry["mMinPotential"]),
        max_clock=MACHINE_MAX_CLOCK,
        is_variable_power=is_variable_power,
        can_change_production_boost=(entry["mCanChangeProductionBoost"] == "True"),
        base_production_boost=float(entry["mBaseProductionBoost"]),
        production_shard_slot_size=production_shard_slot_size,
        production_shard_boost_multiplier=float(
            entry["mProductionShardBoostMultiplier"]
        ),
        production_boost_power_consumption_exponent=float(
            entry["mProductionBoostPowerConsumptionExponent"]
        ),
    )


manufacturers: dict[str, Manufacturer] = {}

for entry in native_class_to_class_entries["FGBuildableManufacturer"]:
    manufacturer = parse_manufacturer(entry, is_variable_power=False)
    manufacturers[manufacturer.class_name] = manufacturer

for entry in native_class_to_class_entries["FGBuildableManufacturerVariablePower"]:
    manufacturer = parse_manufacturer(entry, is_variable_power=True)
    manufacturers[manufacturer.class_name] = manufacturer

debug_dump("Parsed manufacturers", manufacturers)


### Recipes ###


def parse_recipe(entry: dict[str, Any]) -> Recipe | None:
    produced_in = parse_class_list(entry["mProducedIn"]) or []
    recipe_manufacturer = None

    for manufacturer in produced_in:
        if manufacturer in manufacturers:
            recipe_manufacturer = manufacturer
            break

    if recipe_manufacturer is None:
        # check that recipe is not automatable for known reasons
        assert (
            not produced_in
            or "BP_WorkshopComponent_C" in produced_in
            or "BP_BuildGun_C" in produced_in
            or "FGBuildGun" in produced_in
        ), f"{entry["mDisplayName"]} {produced_in}"
        return None

    recipe_rate = 60.0 / float(entry["mManufactoringDuration"])

    def item_rates(key: str):
        return [
            (item, recipe_rate * amount)
            for (item, amount) in find_item_amounts(entry[key])
        ]

    vpc_constant = float(entry["mVariablePowerConsumptionConstant"])
    vpc_factor = float(entry["mVariablePowerConsumptionFactor"])
    # Assuming the mean is exactly halfway for all of the variable power machine types.
    # This appears to be accurate but it's hard to confirm exactly.
    mean_variable_power_consumption = vpc_constant + 0.5 * vpc_factor

    return Recipe(
        class_name=entry["ClassName"],
        display_name=entry["mDisplayName"],
        manufacturer=recipe_manufacturer,
        inputs=item_rates("mIngredients"),
        outputs=item_rates("mProduct"),
        mean_variable_power_consumption=mean_variable_power_consumption,
    )


recipes: dict[str, Recipe] = {}

for entry in native_class_to_class_entries["FGRecipe"]:
    recipe = parse_recipe(entry)
    if recipe is not None:
        recipes[recipe.class_name] = recipe


debug_dump("Parsed recipes", recipes)


### Items ###


def parse_item(entry: dict[str, Any]) -> Item:
    return Item(
        class_name=entry["ClassName"],
        display_name=entry["mDisplayName"],
        form=entry["mForm"],
        points=int(entry["mResourceSinkPoints"]),
        stack_size=STACK_SIZES[entry["mStackSize"]],
        energy=float(entry["mEnergyValue"]),
    )


items: dict[str, Item] = {}


for native_class in ALL_ITEM_NATIVE_CLASSES:
    for entry in native_class_to_class_entries[native_class]:
        item = parse_item(entry)
        items[item.class_name] = item

debug_dump("Parsed items", items)


### Generators ###


def parse_fuel(entry: dict[str, Any]) -> Fuel:
    byproduct_amount = entry["mByproductAmount"]
    return Fuel(
        fuel_class=entry["mFuelClass"],
        supplemental_resource_class=entry["mSupplementalResourceClass"] or None,
        byproduct=entry["mByproduct"] or None,
        byproduct_amount=int(byproduct_amount) if byproduct_amount else 0,
    )


def parse_generator(entry: dict[str, Any]) -> PowerGenerator:
    fuels = [parse_fuel(fuel) for fuel in entry["mFuel"]]

    assert entry["mCanChangePotential"] == "True"
    assert str_to_clock(entry["mMaxPotential"]) == MACHINE_BASE_CLOCK

    return PowerGenerator(
        class_name=entry["ClassName"],
        display_name=entry["mDisplayName"],
        fuels=fuels,
        power_production=float(entry["mPowerProduction"]),
        min_clock=str_to_clock(entry["mMinPotential"]),
        max_clock=MACHINE_MAX_CLOCK,
        requires_supplemental=(entry["mRequiresSupplementalResource"] == "True"),
        supplemental_to_power_ratio=float(entry["mSupplementalToPowerRatio"]),
    )


def parse_geothermal_generator(entry: dict[str, Any]) -> GeothermalGenerator:
    # "mVariablePowerProductionConstant": "0.000000" should be 100 MW, hardcode it here
    vpp_constant = 100.0
    vpp_factor = float(entry["mVariablePowerProductionFactor"])
    # Assuming the mean power production is exactly halfway.
    mean_variable_power_production = vpp_constant + 0.5 * vpp_factor

    assert entry["mCanChangePotential"] == "False"

    return GeothermalGenerator(
        class_name=entry["ClassName"],
        display_name=entry["mDisplayName"],
        min_clock=MACHINE_BASE_CLOCK,
        max_clock=MACHINE_BASE_CLOCK,
        mean_variable_power_production=mean_variable_power_production,
    )


generators: dict[str, PowerGenerator] = {}

for native_class in ALL_GENERATOR_NATIVE_CLASSES:
    for entry in native_class_to_class_entries[native_class]:
        generator = parse_generator(entry)
        generators[generator.class_name] = generator

# geothermal generator (special case)
geothermal_generator = parse_geothermal_generator(
    class_name_to_entry[GEOTHERMAL_GENERATOR_CLASS]
)

debug_dump("Parsed generators", generators)
debug_dump("Parsed geothermal generator", geothermal_generator)


### Map info ###


with open(MAP_INFO_PATH, "r") as f:
    map_info_raw = json.load(f)

map_info: dict[str, Any] = {}

for tab in map_info_raw["options"]:
    if "tabId" in tab:
        map_info[tab["tabId"]] = tab["options"]


### Resources ###


@dataclass
class Resource:
    resource_id: str
    item_class: str
    subtype: str
    multiplier: float
    is_unlimited: bool
    count: int
    is_resource_well: bool
    num_satellites: int


resources: dict[str, Resource] = {}
geysers: dict[str, Resource] = {}


# Persistent_Level:PersistentLevel.BP_FrackingCore6_UAID_40B076DF2F79D3DF01_1961476789
# becomes #6. We can strip out the UAID as long as it's unique for each item type.
FRACKING_CORE_REGEX = re.compile(
    r"Persistent_Level:PersistentLevel\.BP_FrackingCore_?(\d+)(_UAID_\w+)?"
)


def parse_fracking_core_name(s: str) -> str:
    m = FRACKING_CORE_REGEX.fullmatch(s)
    assert m is not None, s
    return "#" + m.group(1)


def parse_and_add_resources(map_resource: dict[str, Any]):
    if "type" not in map_resource:
        return

    item_class = map_resource["type"]
    if item_class in RESOURCE_MAPPINGS:
        item_class = RESOURCE_MAPPINGS[item_class]

    if item_class == GEYSER_CLASS:
        output = geysers
    else:
        output = resources
        assert item_class in items, f"map has unknown resource: {item_class}"

    for node_purity in map_resource["options"]:
        purity = node_purity["purity"]
        nodes = node_purity["markers"]
        if not nodes:
            continue
        sample_node = nodes[0]
        if "core" in sample_node:
            # resource well satellite nodes, map to cores and sum the purity multipliers
            for node in nodes:
                subtype = parse_fracking_core_name(node["core"])
                resource_id = f"{item_class}|{subtype}"
                if resource_id not in output:
                    output[resource_id] = Resource(
                        resource_id=resource_id,
                        item_class=item_class,
                        subtype=subtype,
                        multiplier=0.0,
                        is_unlimited=False,
                        count=1,
                        is_resource_well=True,
                        num_satellites=0,
                    )
                output[resource_id].multiplier += PURITY_MULTIPLIERS[purity]
                output[resource_id].num_satellites += 1
        else:
            # normal nodes, add directly
            subtype = purity  # individual nodes are indistinguishable
            resource_id = f"{item_class}|{subtype}"
            assert resource_id not in output
            output[resource_id] = Resource(
                resource_id=resource_id,
                item_class=item_class,
                subtype=subtype,
                multiplier=PURITY_MULTIPLIERS[purity],
                is_unlimited=False,
                count=len(nodes),
                is_resource_well=False,
                num_satellites=0,
            )


for map_resource in map_info["resource_nodes"]:
    parse_and_add_resources(map_resource)

for map_resource in map_info["resource_wells"]:
    parse_and_add_resources(map_resource)

# Water from extractors is a special infinite resource
resources[f"{WATER_CLASS}|extractor"] = Resource(
    resource_id=f"{WATER_CLASS}|extractor",
    item_class=WATER_CLASS,
    subtype="extractor",
    multiplier=1,
    is_unlimited=True,
    count=0,
    is_resource_well=False,
    num_satellites=0,
)

debug_dump("Parsed resources", resources)
debug_dump("Parsed geysers", geysers)


### Somersloops ###


def find_somersloops_map_layer(
    map_tab_artifacts: list[dict[str, list[dict[str, Any]]]]
):
    for unknown_level in map_tab_artifacts:
        for map_layer in unknown_level["options"]:
            if map_layer["layerId"] == "somersloops":
                return map_layer
    raise RuntimeError("failed to find somersloops map layer")


def parse_num_somersloops_on_map(somersloops_map_layer: dict[str, Any]) -> int:
    count = 0
    for marker in somersloops_map_layer["markers"]:
        if marker["type"] == "somersloop":
            count += 1
    return count


PRODUCTION_AMPLIFIER_UNLOCK_SOMERSLOOP_COST = 1
ALIEN_POWER_AUGMENTER_UNLOCK_SOMERSLOOP_COST = 1
ALIEN_POWER_AUGMENTER_BUILD_SOMERSLOOP_COST = 10


def get_num_somersloops_available() -> int:
    if args.num_somersloops_available is not None:
        return args.num_somersloops_available

    num_somersloops_on_map = parse_num_somersloops_on_map(
        find_somersloops_map_layer(map_info["artifacts"])
    )
    research_somersloop_cost = (
        PRODUCTION_AMPLIFIER_UNLOCK_SOMERSLOOP_COST
        if not args.disable_production_amplification
        else 0
    ) + (
        ALIEN_POWER_AUGMENTER_UNLOCK_SOMERSLOOP_COST
        if args.num_alien_power_augmenters > 0
        else 0
    )
    assert research_somersloop_cost <= num_somersloops_on_map
    return num_somersloops_on_map - research_somersloop_cost


NUM_SOMERSLOOPS_AVAILABLE = get_num_somersloops_available()
POWER_SOMERSLOOP_COST: int = (
    ALIEN_POWER_AUGMENTER_BUILD_SOMERSLOOP_COST * args.num_alien_power_augmenters
)

assert (
    POWER_SOMERSLOOP_COST <= NUM_SOMERSLOOPS_AVAILABLE
), f"{POWER_SOMERSLOOP_COST=} {NUM_SOMERSLOOPS_AVAILABLE=}"

NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION = (
    NUM_SOMERSLOOPS_AVAILABLE - POWER_SOMERSLOOP_COST
)

assert args.num_fueled_alien_power_augmenters <= args.num_alien_power_augmenters

ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER: float = (
    ALIEN_POWER_AUGMENTER_STATIC_POWER * args.num_alien_power_augmenters
)
ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST: float = (
    ALIEN_POWER_AUGMENTER_BASE_CIRCUIT_BOOST
    * (args.num_alien_power_augmenters - args.num_fueled_alien_power_augmenters)
) + (
    ALIEN_POWER_AUGMENTER_FUELED_CIRCUIT_BOOST * args.num_fueled_alien_power_augmenters
)
POWER_PRODUCTION_MULTIPLIER = 1.0 + ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST
TOTAL_ALIEN_POWER_MATRIX_COST: float = (
    ALIEN_POWER_AUGMENTER_FUEL_INPUT_RATE * args.num_fueled_alien_power_augmenters
)

debug_dump(
    "Somersloops",
    f"""
{args.disable_production_amplification=}
{args.num_alien_power_augmenters=}
{args.num_fueled_alien_power_augmenters=}
{NUM_SOMERSLOOPS_AVAILABLE=}
{NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION=}
{ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER=}
{ALIEN_POWER_AUGMENTER_TOTAL_CIRCUIT_BOOST=}
{POWER_PRODUCTION_MULTIPLIER=}
{TOTAL_ALIEN_POWER_MATRIX_COST=}
""".strip(),
)


### Additional helpers ###


def get_power_consumption(
    machine: PowerConsumer, clock: Fraction, recipe: Recipe | None = None
) -> float:
    power_consumption = machine.power_consumption
    if recipe is not None and machine.is_variable_power:
        power_consumption += recipe.mean_variable_power_consumption
    return power_consumption * (clock**machine.power_consumption_exponent)


def get_power_production(generator: PowerGenerator, clock: Fraction) -> float:
    return generator.power_production * clock


def get_miner_for_resource(resource: Resource) -> Miner:
    item_class = resource.item_class
    item = items[item_class]
    candidates: list[Miner] = []
    for miner in miners.values():
        if (
            miner.uses_resource_wells == resource.is_resource_well
            and miner.check_allowed_resource(item_class, item.form)
        ):
            candidates.append(miner)
    assert candidates, f"could not find miner for {item_class}"
    assert len(candidates) == 1, f"more than one miner for {item_class}: {candidates}"
    return candidates[0]


def get_form_conveyance_limit(form: str) -> float:
    if form == "RF_SOLID":
        return CONVEYOR_BELT_LIMIT
    elif form == "RF_LIQUID" or form == "RF_GAS":
        return PIPELINE_LIMIT
    else:
        assert False


def get_conveyance_limit_clock(item: Item, rate: float) -> Fraction:
    conveyance_limit = get_form_conveyance_limit(item.form)
    return float_to_clock(conveyance_limit / rate)


def get_max_extraction_clock(
    miner: Miner, resource: Resource, extraction_rate: float
) -> Fraction:
    max_clock = miner.max_clock

    # Assume individual Resource Well Extractors can never exceed conveyance limit
    if resource.is_resource_well:
        return max_clock

    item = items[resource.item_class]
    return min(max_clock, get_conveyance_limit_clock(item, extraction_rate))


def get_max_recipe_clock(
    machine: Machine, recipe: Recipe, output_multiplier: float = 1.0
) -> Fraction:
    max_clock = machine.max_clock

    for item_class, input_rate in recipe.inputs:
        max_clock = min(
            max_clock,
            get_conveyance_limit_clock(items[item_class], input_rate),
        )

    for item_class, output_rate in recipe.outputs:
        max_clock = min(
            max_clock,
            get_conveyance_limit_clock(
                items[item_class], output_rate * output_multiplier
            ),
        )

    return max_clock


def clamp_clock_choices(
    configured_clocks: list[Fraction], min_clock: Fraction, max_clock: Fraction
) -> list[Fraction]:
    assert min_clock < max_clock
    return sorted(
        {min(max_clock, max(min_clock, clock)) for clock in configured_clocks}
    )


# It's convenient to consider generators burning fuel as recipes,
# even though they are not actually listed as recipes.
def create_recipe_for_generator(generator: PowerGenerator, fuel: Fuel) -> Recipe:
    inputs: list[tuple[str, float]] = []
    outputs: list[tuple[str, float]] = []

    power_production = generator.power_production

    fuel_class = fuel.fuel_class
    fuel_item = items[fuel_class]
    fuel_rate = 60.0 * power_production / fuel_item.energy
    inputs.append((fuel_class, fuel_rate))

    if generator.requires_supplemental:
        assert fuel.supplemental_resource_class is not None
        supplemental_class = fuel.supplemental_resource_class
        supplemental_rate = (
            60.0 * power_production * generator.supplemental_to_power_ratio
        )
        inputs.append((supplemental_class, supplemental_rate))

    if fuel.byproduct is not None:
        byproduct_class = fuel.byproduct
        byproduct_rate = fuel_rate * fuel.byproduct_amount
        outputs.append((byproduct_class, byproduct_rate))

    return Recipe(
        class_name="",
        display_name="",
        manufacturer=generator.class_name,
        inputs=inputs,
        outputs=outputs,
        mean_variable_power_consumption=0.0,
    )


def get_item_display_name(item_class: str) -> str:
    if item_class in items:
        return items[item_class].display_name
    else:
        return ADDITIONAL_DISPLAY_NAMES[item_class]


### LP setup ###


@dataclass
class LPColumn:
    coeffs: dict[str, float]
    type_: str
    name: str
    display_name: str
    full_display_name: str
    machine_name: str | None
    resource_subtype: str | None
    clock: Fraction | None
    somersloops: int | None
    objective_weight: float | None
    requires_integrality: bool


lp_columns: dict[str, LPColumn] = {}
lp_equalities: dict[str, float] = {}
lp_lower_bounds: dict[str, float] = {}


def create_column_id(
    type_: str,
    name: str,
    clock: Fraction | None = None,
    somersloops: int | None = None,
):
    tokens = [type_, name]
    if clock is not None:
        tokens.append(clock_to_percent_str(clock))
    if somersloops is not None:
        tokens.append(f"S:{somersloops}")
    return "|".join(tokens)


def create_column_full_display_name(
    type_: str,
    display_name: str,
    machine_name: str | None,
    resource_subtype: str | None,
    clock: Fraction | None = None,
    somersloops: int | None = None,
):
    tokens = [machine_name or type_, display_name]
    if resource_subtype is not None:
        tokens.append(resource_subtype)
    if clock is not None:
        tokens.append(clock_to_percent_str(clock))
    if somersloops is not None:
        tokens.append(f"S:{somersloops}")
    return "|".join(tokens)


def add_lp_column(
    coeffs: dict[str, float],
    type_: str,
    name: str,
    display_name: str | None = None,
    machine_name: str | None = None,
    resource_subtype: str | None = None,
    clock: Fraction | None = None,
    somersloops: int | None = None,
    objective_weight: float | None = None,
    requires_integrality: bool = False,
):
    column_id = create_column_id(
        type_=type_,
        name=name,
        clock=clock,
        somersloops=somersloops,
    )
    display_name = display_name or name
    full_display_name = create_column_full_display_name(
        type_=type_,
        display_name=display_name,
        machine_name=machine_name,
        resource_subtype=resource_subtype,
        clock=clock,
        somersloops=somersloops,
    )
    assert column_id not in lp_columns, f"duplicate {column_id=}"
    lp_columns[column_id] = LPColumn(
        coeffs=coeffs,
        type_=type_,
        name=name,
        display_name=display_name,
        full_display_name=full_display_name,
        machine_name=machine_name,
        resource_subtype=resource_subtype,
        clock=clock,
        somersloops=somersloops,
        objective_weight=objective_weight,
        requires_integrality=requires_integrality,
    )


def get_recipe_coeffs(
    recipe: Recipe, clock: Fraction, output_multiplier: float = 1.0
) -> defaultdict[str, float]:
    coeffs: defaultdict[str, float] = defaultdict(float)

    for item_class, input_rate in recipe.inputs:
        item_var = f"item|{item_class}"
        coeffs[item_var] -= clock * input_rate

    for item_class, output_rate in recipe.outputs:
        item_var = f"item|{item_class}"
        coeffs[item_var] += clock * output_rate * output_multiplier

    return coeffs


def add_miner_columns(resource: Resource):
    resource_id = resource.resource_id
    item_class = resource.item_class
    item = items[item_class]

    miner = get_miner_for_resource(resource)

    extraction_rate = miner.extraction_rate_base * resource.multiplier
    min_clock = miner.min_clock
    max_clock = get_max_extraction_clock(miner, resource, extraction_rate)
    configured_clocks = MANUFACTURER_CLOCKS if resource.is_unlimited else MINER_CLOCKS
    clock_choices = clamp_clock_choices(configured_clocks, min_clock, max_clock)

    resource_var = f"resource|{resource_id}"
    item_var = f"item|{item_class}"

    for clock in clock_choices:
        machines = 1 + (resource.num_satellites if resource.is_resource_well else 0)
        coeffs = {
            "machines": machines,
            "power_consumption": get_power_consumption(miner, clock),
            item_var: clock * extraction_rate,
        }

        if not resource.is_unlimited:
            coeffs[resource_var] = -1

        add_lp_column(
            coeffs,
            type_="miner",
            name=resource_id,
            display_name=item.display_name,
            machine_name=miner.display_name,
            resource_subtype=resource.subtype,
            clock=clock,
        )

    if not resource.is_unlimited:
        resource_multiplier = RESOURCE_MULTIPLIERS.get(resource.item_class, 1.0)
        lp_lower_bounds[resource_var] = -resource.count * resource_multiplier

    lp_equalities[item_var] = 0.0


for resource in resources.values():
    add_miner_columns(resource)


def add_manufacturer_columns(recipe: Recipe):
    manufacturer_class = recipe.manufacturer
    manufacturer = manufacturers[manufacturer_class]

    somersloop_choices: list[int | None] = [None]
    if manufacturer.can_change_production_boost:
        somersloop_choices.extend(range(1, manufacturer.production_shard_slot_size + 1))

    for somersloops in somersloop_choices:
        if somersloops is None:
            output_multiplier = 1.0
            power_multiplier = 1.0
            requires_integrality = False
        else:
            output_multiplier = (
                1.0 + somersloops * manufacturer.production_shard_boost_multiplier
            )
            power_multiplier = (
                output_multiplier
                ** manufacturer.production_boost_power_consumption_exponent
            )
            # Fractional machines are generally fine due to clock speeds,
            # but we should not allow fractional somersloops.
            requires_integrality = True

        min_clock = manufacturer.min_clock
        max_clock = get_max_recipe_clock(
            manufacturer, recipe, output_multiplier=output_multiplier
        )
        configured_clocks = (
            MANUFACTURER_CLOCKS if somersloops is None else SOMERSLOOP_CLOCKS
        )
        clock_choices = clamp_clock_choices(configured_clocks, min_clock, max_clock)

        for clock in clock_choices:
            power_consumption = (
                get_power_consumption(manufacturer, clock, recipe) * power_multiplier
            )
            coeffs = {
                "machines": 1,
                "power_consumption": power_consumption,
            }
            if somersloops is not None:
                coeffs["somersloop_usage"] = somersloops

            recipe_coeffs = get_recipe_coeffs(
                recipe, clock=clock, output_multiplier=output_multiplier
            )
            for item_var, coeff in recipe_coeffs.items():
                coeffs[item_var] = coeff
                lp_equalities[item_var] = 0.0

            add_lp_column(
                coeffs,
                type_="manufacturer",
                name=recipe.class_name,
                display_name=recipe.display_name,
                machine_name=manufacturer.display_name,
                clock=clock,
                somersloops=somersloops,
                requires_integrality=requires_integrality,
            )


for recipe in recipes.values():
    if recipe.class_name not in DISABLED_RECIPES:
        add_manufacturer_columns(recipe)


def add_sink_column(item: Item):
    item_class = item.class_name
    item_var = f"item|{item_class}"
    points = item.points

    if not (item.form == "RF_SOLID" and points > 0):
        if args.allow_waste:
            add_lp_column(
                {item_var: -1},
                type_="waste",
                name=item_class,
                display_name=item.display_name,
            )
        return

    coeffs = {
        "machines": 1 / CONVEYOR_BELT_LIMIT,
        "power_consumption": SINK_POWER_CONSUMPTION / CONVEYOR_BELT_LIMIT,
        item_var: -1,
        "points": points,
    }

    add_lp_column(
        coeffs,
        type_="sink",
        name=item_class,
        display_name=item.display_name,
    )

    lp_equalities[item_var] = 0.0


for item in items.values():
    add_sink_column(item)


def add_generator_columns(generator: PowerGenerator, fuel: Fuel):
    recipe = create_recipe_for_generator(generator, fuel)
    fuel_item = items[fuel.fuel_class]

    min_clock = generator.min_clock
    max_clock = get_max_recipe_clock(generator, recipe)
    clock_choices = clamp_clock_choices(GENERATOR_CLOCKS, min_clock, max_clock)

    for clock in clock_choices:
        power_production = (
            get_power_production(generator, clock=clock) * POWER_PRODUCTION_MULTIPLIER
        )
        coeffs = {
            "machines": 1,
            "power_production": power_production,
        }

        recipe_coeffs = get_recipe_coeffs(recipe, clock=clock)
        for item_var, coeff in recipe_coeffs.items():
            coeffs[item_var] = coeff
            lp_equalities[item_var] = 0.0

        add_lp_column(
            coeffs,
            type_="generator",
            name=fuel_item.class_name,
            display_name=fuel_item.display_name,
            machine_name=generator.display_name,
            clock=clock,
        )


for generator in generators.values():
    for fuel in generator.fuels:
        add_generator_columns(generator, fuel)


def add_geothermal_generator_columns(resource: Resource):
    resource_id = resource.resource_id
    resource_var = f"resource|{resource_id}"

    power_production = (
        geothermal_generator.mean_variable_power_production
        * resource.multiplier
        * POWER_PRODUCTION_MULTIPLIER
    )
    coeffs = {
        "machines": 1,
        "power_production": power_production,
        resource_var: -1,
    }

    add_lp_column(
        coeffs,
        type_="generator",
        name=resource_id,
        display_name=get_item_display_name(GEYSER_CLASS),
        machine_name=geothermal_generator.display_name,
        resource_subtype=resource.subtype,
        requires_integrality=True,
    )

    resource_multiplier = RESOURCE_MULTIPLIERS.get(resource.item_class, 1.0)
    lp_lower_bounds[resource_var] = -resource.count * resource_multiplier


for resource in geysers.values():
    add_geothermal_generator_columns(resource)


def add_meta_coeffs(column_id: str, column: LPColumn):
    to_add: defaultdict[str, float] = defaultdict(float)
    for variable, coeff in column.coeffs.items():
        if variable.startswith("item|") and coeff > 0:
            item_class = variable[5:]
            if item_class not in items:
                print(f"WARNING: item not found in items dict: {item_class}")
                continue

            item = items[item_class]
            form = item.form
            conveyance_limit = get_form_conveyance_limit(form)
            conveyance = coeff / conveyance_limit

            # Avoid incurring transport costs for Water Extractors,
            # as they would otherwise dominate the cost.
            # Basically we're assuming other stuff is brought to the water.
            if column.type_ == "miner" and column.resource_subtype != "extractor":
                to_add["transport_power_cost"] += args.transport_power_cost * conveyance
                to_add["drone_battery_cost"] += args.drone_battery_cost * conveyance

            if form == "RF_SOLID":
                to_add["conveyors"] += conveyance
            else:
                to_add["pipelines"] += conveyance

    for variable, coeff in to_add.items():
        if coeff != 0.0:
            column.coeffs[variable] = column.coeffs.get(variable, 0.0) + coeff


for column_id, column in lp_columns.items():
    add_meta_coeffs(column_id, column)


@dataclass
class HardLimit:
    name: str
    weight: float
    lower_bound: float


machine_limit = (
    HardLimit(name="machine_limit", weight=-1.0, lower_bound=-args.machine_limit)
    if args.machine_limit is not None
    else None
)


def add_objective_column(
    objective: str, objective_weight: float, hard_limit: HardLimit | None = None
):
    coeffs = {
        objective: -1.0,
    }
    if hard_limit is not None:
        coeffs[hard_limit.name] = hard_limit.weight
        lp_lower_bounds[hard_limit.name] = hard_limit.lower_bound
    add_lp_column(
        coeffs,
        type_="objective",
        name=objective,
        objective_weight=objective_weight,
    )
    lp_equalities[objective] = 0.0


add_objective_column("points", 1.0)
add_objective_column("machines", -args.machine_penalty, machine_limit)
add_objective_column("conveyors", -args.conveyor_penalty)
add_objective_column("pipelines", -args.pipeline_penalty)


# These columns cancel dummy variables introduced for ease of reporting.
# Instead of deducting X directly, we accumulate a cost variable, then pay it here.
# Breakdowns of cost contributors/payers then appear naturally in the report.

# Power usage
coeffs = {
    "power_consumption": -1.0,
    "power_production": -1.0,
}
add_lp_column(
    coeffs,
    type_="power",
    name="usage",
)
lp_equalities["power_consumption"] = 0.0
lp_lower_bounds["power_production"] = -ALIEN_POWER_AUGMENTER_TOTAL_STATIC_POWER
if args.infinite_power:
    lp_lower_bounds["power_production"] = -np.inf

# Alien Power Matrix fuel
if TOTAL_ALIEN_POWER_MATRIX_COST > 0.0:
    coeffs = {
        "alien_power_matrix_cost": -1.0,
        f"item|{ALIEN_POWER_MATRIX_CLASS}": -1.0,
    }
    add_lp_column(
        coeffs,
        type_="alien_power_matrix",
        name="fuel",
    )
    lp_equalities["alien_power_matrix_cost"] = -TOTAL_ALIEN_POWER_MATRIX_COST

# Configured extra costs
for extra_cost, cost_variable, cost_coeff in [
    ("transport_power_cost", "power_consumption", 1.0),
    ("drone_battery_cost", f"item|{BATTERY_CLASS}", -1.0),
]:
    coeffs = {
        extra_cost: -1.0,
        cost_variable: cost_coeff,
    }
    add_lp_column(
        coeffs,
        type_="extra_cost",
        name=extra_cost,
    )
    lp_equalities[extra_cost] = 0.0

# Somersloop usage
coeffs = {
    "somersloop_usage": -1.0,
    "somersloop": -1.0,
}
add_lp_column(
    coeffs,
    type_="somersloop",
    name="usage",
)
lp_equalities["somersloop_usage"] = 0.0
lp_lower_bounds["somersloop"] = -NUM_SOMERSLOOPS_AVAILABLE_FOR_PRODUCTION

# debug_dump("LP columns (before pruning)", lp_columns)
# debug_dump("LP equalities (before pruning)", lp_equalities)
# debug_dump("LP lower bounds (before pruning)", lp_lower_bounds)


def get_all_variables() -> set[str]:
    variables: set[str] = set()

    for column in lp_columns.values():
        for variable in column.coeffs.keys():
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


lp_variables = get_all_variables()

# debug_dump("LP variables (before pruning)", lp_variables)


### Pruning unreachable items ###


reachable_items: set[str] = set()
while True:
    any_added = False
    for column_id, column in lp_columns.items():
        eligible = True
        to_add: set[str] = set()
        for variable, coeff in column.coeffs.items():
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

unreachable_items = (
    set(v for v in lp_variables if v.startswith("item|")) - reachable_items
)

debug_dump("Unreachable items to be pruned", unreachable_items)

columns_to_prune: list[str] = []
for column_id, column in lp_columns.items():
    for variable, coeff in column.coeffs.items():
        if variable in unreachable_items and coeff < 0:
            columns_to_prune.append(column_id)
            break
for column_id in columns_to_prune:
    del lp_columns[column_id]
for item_var in unreachable_items:
    if item_var in lp_equalities:
        del lp_equalities[item_var]


debug_dump("LP columns (after pruning)", lp_columns)
debug_dump("LP equalities (after pruning)", lp_equalities)
debug_dump("LP lower bounds (after pruning)", lp_lower_bounds)

lp_variables = get_all_variables()

debug_dump("LP variables (after pruning)", lp_variables)


### LP run ###


def to_index_map(seq: Iterable[T]) -> dict[T, int]:
    return {value: index for index, value in enumerate(seq)}


def from_index_map(d: dict[T, int]) -> list[T]:
    result: list[T | None] = [None] * len(d)
    for value, index in d.items():
        result[index] = value
    assert all(value is not None for value in result)
    return cast(list[T], result)


# order is for report display, but we might as well sort it here
column_type_order = to_index_map(
    [
        "objective",
        "power",
        "extra_cost",
        "sink",
        "waste",
        "somersloop",
        "manufacturer",
        "miner",
        "generator",
    ]
)
resource_subtype_order = to_index_map(["pure", "normal", "impure"])
objective_order = to_index_map(
    ["points", "machines", "machine_limit", "conveyors", "pipelines"]
)
extra_cost_order = to_index_map(["transport_power_cost", "drone_battery_cost"])


def column_order_key(arg: tuple[str, LPColumn]):
    column_id, column = arg

    if column.type_ in column_type_order:
        type_key = (0, column_type_order[column.type_])
    else:
        type_key = (1, column.type_)

    name = column.display_name
    if column.type_ == "objective":
        name_key = objective_order[name]
    elif column.type_ == "extra_cost":
        name_key = extra_cost_order[name]
    else:
        name_key = name

    resource_subtype = column.resource_subtype
    if resource_subtype in resource_subtype_order:
        subtype_key = (0, resource_subtype_order[resource_subtype])
    else:
        subtype_key = (1, resource_subtype)

    return (type_key, name_key, subtype_key, column.clock, column_id)


sorted_columns = sorted(lp_columns.items(), key=column_order_key)
lp_variable_indices = to_index_map(lp_variables)

lp_c = np.zeros(len(lp_columns), dtype=np.double)
lp_integrality = np.zeros(len(lp_columns), dtype=np.int64)
lp_A = np.zeros((len(lp_variables), len(lp_columns)), dtype=np.double)
lp_b_l = np.zeros(len(lp_variables), dtype=np.double)
lp_b_u = np.zeros(len(lp_variables), dtype=np.double)

for column_index, (column_id, column) in enumerate(sorted_columns):
    if column.objective_weight is not None:
        lp_c[column_index] = column.objective_weight
    if column.requires_integrality:
        lp_integrality[column_index] = 1
    for variable, coeff in column.coeffs.items():
        lp_A[lp_variable_indices[variable], column_index] = coeff

for variable, rhs in lp_equalities.items():
    lp_b_l[lp_variable_indices[variable]] = rhs
    lp_b_u[lp_variable_indices[variable]] = rhs

for variable, rhs in lp_lower_bounds.items():
    lp_b_l[lp_variable_indices[variable]] = rhs

lp_constraints = scipy.optimize.LinearConstraint(lp_A, lp_b_l, lp_b_u)  # type: ignore

print("LP running")

lp_result = scipy.optimize.milp(
    -lp_c,
    integrality=lp_integrality,
    constraints=lp_constraints,
)

if lp_result.status != 0:
    print("ERROR: LP did not terminate successfully")
    pprint(lp_result)
    sys.exit(1)

print("LP result:")
pprint(lp_result)


### Display formatting ###


REPORT_EPSILON = 1e-7

column_results: list[tuple[str, LPColumn, float]] = [
    (column_id, column, lp_result.x[column_index])
    for column_index, (column_id, column) in enumerate(sorted_columns)
]

if not args.show_unused:
    column_results = list(filter(lambda x: abs(x[2]) > REPORT_EPSILON, column_results))


@dataclass
class BudgetEntry:
    desc: str
    count: float
    rate: float
    share: float


@dataclass
class VariableBreakdown:
    type_: str
    display_name: str
    sort_key: Any
    production: list[BudgetEntry]
    consumption: list[BudgetEntry]
    initial: float | None
    final: float | None


variable_type_order = to_index_map(
    from_index_map(objective_order)
    + ["power_production", "power_consumption"]
    + ["alien_power_matrix_cost"]
    + from_index_map(extra_cost_order)
    + ["somersloop", "somersloop_usage"]
    + ["item", "resource"]
)


def create_empty_variable_breakdown(variable: str) -> VariableBreakdown:
    tokens = variable.split("|")
    type_ = tokens[0]
    if type_ == "item" or type_ == "resource":
        item_class = tokens[1]
        tokens[1] = get_item_display_name(item_class)
    display_name = "|".join(tokens)
    sort_key: list[Any] = [variable_type_order[type_]]
    if type_ == "resource":
        sort_key.append(tokens[1])
        sort_key.append(resource_subtype_order.get(tokens[2], np.inf))
        sort_key.append(tokens[2])
    else:
        sort_key.append(display_name)
    return VariableBreakdown(
        type_=type_,
        display_name=display_name,
        sort_key=sort_key,
        production=[],
        consumption=[],
        initial=None,
        final=None,
    )


variable_breakdowns = {
    variable: create_empty_variable_breakdown(variable) for variable in lp_variables
}

lp_objective: float = -lp_result.fun

print("")
print("Summary:")

print(f"{lp_objective:>17.3f} objective")

for column_id, column, column_coeff in column_results:
    print(f"{column_coeff:>17.3f} {column.full_display_name}")

    for variable, coeff in column.coeffs.items():
        rate = column_coeff * coeff
        budget_entry = BudgetEntry(
            desc=column.full_display_name,
            count=column_coeff,
            rate=abs(rate),
            share=0.0,
        )
        if abs(rate) < REPORT_EPSILON:
            continue
        elif rate > 0:
            variable_breakdowns[variable].production.append(budget_entry)
        else:
            variable_breakdowns[variable].consumption.append(budget_entry)

print("")


def finalize_variable_budget_side(budget_side: list[BudgetEntry]):
    if not budget_side:
        return
    total_rate = 0.0
    total_count = 0.0
    for budget_entry in budget_side:
        total_rate += budget_entry.rate
        total_count += budget_entry.count
    for budget_entry in budget_side:
        budget_entry.share = budget_entry.rate / total_rate
    budget_side.sort(key=lambda entry: (-entry.share, entry.desc))
    total = BudgetEntry(desc="Total", count=total_count, rate=total_rate, share=1.0)
    budget_side.insert(0, total)


lp_Ax = lp_A @ lp_result.x

for variable, breakdown in variable_breakdowns.items():
    # don't show offsetting dummy items in the breakdown (e.g. "objective|points" as consumer of points)
    # currently these are precisely the consumption of special variables, but that may change
    if breakdown.type_ not in ["item", "resource"]:
        breakdown.consumption = []

    variable_index = lp_variable_indices[variable]
    if variable in lp_lower_bounds:
        slack: float = lp_Ax[variable_index] - lp_b_l[variable_index]
        if slack < -REPORT_EPSILON:
            print(f"WARNING: lower bound violation: {variable=} {slack=}")
        breakdown.initial = -lp_lower_bounds[variable]
        breakdown.final = slack if abs(slack) > REPORT_EPSILON else 0
    else:
        residual: float = lp_Ax[variable_index] - lp_b_l[variable_index]
        if abs(residual) > REPORT_EPSILON:
            print(f"WARNING: equality constraint violation: {variable=} {residual=}")
    finalize_variable_budget_side(breakdown.production)
    finalize_variable_budget_side(breakdown.consumption)

sorted_variable_breakdowns = sorted(
    variable_breakdowns.values(), key=lambda bd: bd.sort_key
)

if args.xlsx_report:
    print("Writing xlsx report")

    import xlsxwriter

    workbook = xlsxwriter.Workbook(args.xlsx_report, {"nan_inf_to_errors": True})

    default_format = workbook.add_format({"align": "center"})
    top_format = workbook.add_format({"align": "center", "top": True})
    bold_format = workbook.add_format({"align": "center", "bold": True})
    bold_underline_format = workbook.add_format(
        {"align": "center", "bold": True, "underline": True}
    )
    bold_top_format = workbook.add_format(
        {"align": "center", "bold": True, "top": True}
    )
    bold_underline_top_format = workbook.add_format(
        {"align": "center", "bold": True, "underline": True, "top": True}
    )
    percent_format = workbook.add_format({"align": "center", "num_format": "0.0#####%"})

    sheet_breakdown = workbook.add_worksheet("Breakdown" + args.xlsx_sheet_suffix)
    sheet_list = workbook.add_worksheet("List" + args.xlsx_sheet_suffix)
    sheet_config = workbook.add_worksheet("Config" + args.xlsx_sheet_suffix)

    def write_cell(sheet, *args, fmt=default_format):
        sheet.write(*args, fmt)

    sheet_list.add_table(
        0,
        0,
        len(column_results),
        6,
        {
            "columns": [
                {"header": header, "header_format": bold_format}
                for header in [
                    "Type",
                    "Name",
                    "Machine",
                    "Subtype",
                    "Clock",
                    "Somersloops",
                    "Quantity",
                ]
            ],
            "style": "Table Style Light 16",
        },
    )

    write_cell(sheet_list, 1, 0, "objective")
    write_cell(sheet_list, 1, 1, "objective")
    write_cell(sheet_list, 1, 6, lp_objective)

    for i, (column_id, column, column_coeff) in enumerate(column_results):
        write_cell(sheet_list, 2 + i, 0, column.type_)
        write_cell(sheet_list, 2 + i, 1, column.display_name)
        write_cell(sheet_list, 2 + i, 2, column.machine_name)
        write_cell(sheet_list, 2 + i, 3, column.resource_subtype)
        write_cell(sheet_list, 2 + i, 4, column.clock, fmt=percent_format)
        write_cell(sheet_list, 2 + i, 5, column.somersloops)
        write_cell(sheet_list, 2 + i, 6, column_coeff)

    for c, width in enumerate([19, 39, 25, 19, 11, 17, 13]):
        sheet_list.set_column(c, c, width)

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
        "max_color": "#99FF99",
    }
    consumption_share_cf = production_share_cf.copy()
    consumption_share_cf["max_color"] = "#FFCC66"

    for variable_index, breakdown in enumerate(sorted_variable_breakdowns):
        for budget_side_index, budget_side_name, budget_side in (
            (0, "production", breakdown.production),
            (1, "consumption", breakdown.consumption),
        ):
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
                write_cell(
                    sheet_breakdown, current_row, 0, breakdown.display_name, fmt=fmts[0]
                )
                write_cell(sheet_breakdown, current_row, 1, name, fmt=fmts[0])
                for i, entry in enumerate(budget_side):
                    value = getattr(entry, key)
                    write_cell(sheet_breakdown, current_row, 2 + i, value, fmt=fmts[1])
                if key == "share":
                    cf = (
                        production_share_cf
                        if budget_side_name == "production"
                        else consumption_share_cf
                    )
                    sheet_breakdown.conditional_format(
                        current_row, 3, current_row, len(budget_side) + 1, cf
                    )
                max_budget_entries = max(max_budget_entries, len(budget_side))
                current_row += 1

        for initial_or_final, initial_or_final_value in (
            ("initial", breakdown.initial),
            ("final", breakdown.final),
        ):
            if initial_or_final_value is None:
                continue
            if initial_or_final == "initial":
                fmts = (bold_top_format, top_format)
            else:
                fmts = (bold_format, default_format)
            write_cell(
                sheet_breakdown, current_row, 0, breakdown.display_name, fmt=fmts[0]
            )
            write_cell(
                sheet_breakdown,
                current_row,
                1,
                initial_or_final.capitalize(),
                fmt=fmts[0],
            )
            write_cell(
                sheet_breakdown, current_row, 2, initial_or_final_value, fmt=fmts[1]
            )
            current_row += 1

    for c, width in enumerate([41, 19, 13] + [59] * (max_budget_entries - 1)):
        sheet_breakdown.set_column(c, c, width)

    for i, (arg_name, arg_value) in enumerate(vars(args).items()):
        write_cell(sheet_config, i, 0, arg_name)
        write_cell(sheet_config, i, 1, arg_value)

    for c, width in enumerate([36, 19]):
        sheet_config.set_column(c, c, width)

    workbook.close()
