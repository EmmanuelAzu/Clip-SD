"""Single source of truth for the curated evaluation subset and pruning
schedule, shared across every script in the (now streamlined) pipeline.

Replaces the earlier full-taxonomy (77-class, ~7,200-image balanced)
evaluation scope. The dominant compute cost throughout this pipeline has
always been re-encoding the candidate image pool through the vision
encoder at every pruning level -- restricting that pool to a small,
curated, taxonomically well-spread set of classes cuts that cost by
roughly two orders of magnitude, which matters far more than any other
lever (pruning schedule granularity, scenario count, etc).

CURATED_CLASSES: 10 well-known, easily-recognizable specific classes, one
per coordinate (basic-level) group for maximum taxonomic spread -- 8
animals (each from a different coordinate group: Domestic Dogs, Felines,
Primates, Birds, Reptiles, Amphibians, Fish, Insects) + 2 plants (Citrus,
Fruits). All 10 are already downloaded and verified against
data/taxonomy_full.json.
"""

import numpy as np

CURATED_CLASSES = [
    "German Shepherd",     # Domestic Dogs
    "Tabby Cat",            # Felines
    "Chimpanzee",           # Primates
    "King Penguin",         # Birds (atypical -- flightless, useful contrast)
    "American Alligator",   # Reptiles
    "Bullfrog",              # Amphibians
    "Goldfish",               # Fish
    "Monarch Butterfly",    # Insects
    "Lemon",                  # Plants: Citrus
    "Banana",                 # Plants: Fruits
]

# 2.5% increments, 0% to 70% inclusive (29 levels). Concentrates resolution
# in the region where the real pruning sweep showed the interesting
# hierarchical transitions happening (roughly 20-60%), and drops the
# 70-90% tail, which prior real results showed was already flat at floor
# performance -- i.e. redundant data points that cost compute without
# adding information.
PRUNING_LEVELS_FOCUSED = [round(x, 3) for x in np.arange(0.00, 0.701, 0.025).tolist()]
