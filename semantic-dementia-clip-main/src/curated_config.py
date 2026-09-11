"""Single source of truth for the curated evaluation subset, color scheme,
and pruning schedule, shared across every script in the pipeline.

REDESIGNED from the previous version, which picked exactly ONE specific
breed per coordinate group. That design could not test the project's
central hypothesis at all: with only one breed per group, "coordinate
error" (confusing two breeds within the same group, e.g. German Shepherd
vs Golden Retriever) was structurally impossible -- every wrong answer had
to jump straight to a different species. The whole point of this research
(RQ1: fine-grained distinctions collapse before broad categories) requires
multiple breeds/species per coordinate group so that within-group collapse
is observable at all, and observable as happening BEFORE between-group
collapse.

CURATED_TAXONOMY: 9 coordinate groups, each with 2-6 specific
breeds/species (33 specific classes total), weighted heavily toward
Animals/Plants (94%) with one small Non-Living group (Kitchen Objects) for
contrast. Every entry verified against data/taxonomy_full.json and
confirmed already downloaded (excludes the 5 classes -- Hen, Turtle,
Daisy, Cucumber, Lawnmower -- that don't exist in Tiny-ImageNet-200).
"""

import numpy as np

CURATED_TAXONOMY: dict[str, list[str]] = {
    "Domestic Dogs": [
        "German Shepherd", "Golden Retriever", "Labrador Retriever",
        "Chihuahua", "Standard Poodle", "Yorkshire Terrier",
    ],
    "Felines": ["Tabby Cat", "Egyptian Cat", "Persian Cat", "Cougar", "Lion"],
    "Insects": ["Monarch Butterfly", "Sulphur Butterfly", "Ladybug", "Dragonfly", "Mantis"],
    "Birds": ["Albatross", "Goose", "Black Stork", "King Penguin"],
    "Primates": ["Chimpanzee", "Orangutan", "Baboon"],
    "Amphibians": ["Bullfrog", "Tailed Frog", "European Fire Salamander"],
    "Citrus": ["Lemon", "Orange"],
    "Fruits": ["Banana", "Pomegranate", "Acorn"],
    "Kitchen Objects": ["Frying Pan", "Teapot"],
}

# Flat list, for code that just needs "is this class in scope" rather than
# the group structure (e.g. metadata filtering).
CURATED_CLASSES: list[str] = [c for group in CURATED_TAXONOMY.values() for c in group]

# Fixed 5 images per specific class, regardless of how many are actually
# available (some classes have 100-200 downloaded) -- this is intentional:
# the point is to study collapse ORDER (within-group before between-group)
# on a fast, small, evenly-weighted sample, not to maximize statistical
# power per class.
IMAGES_PER_CLASS = 5

# One base sequential colormap per coordinate group ("hue family"), with
# each specific breed/species within that group assigned a distinct shade
# from the family via COORDINATE_COLOR_SHADES below. This makes visual
# proximity in a plot legend map onto taxonomic proximity: two shades of
# blue are both dogs; blue vs. orange are different coordinate groups
# entirely. Families chosen to be maximally distinguishable from each
# other at a glance.
COORDINATE_CMAP_FAMILIES: dict[str, str] = {
    "Domestic Dogs": "Blues",
    "Felines": "Oranges",
    "Insects": "RdPu",
    "Birds": "Greens",
    "Primates": "Purples",
    "Amphibians": "YlOrBr",
    "Citrus": "YlGn",
    "Fruits": "PuRd",
    "Kitchen Objects": "Greys",
}


def build_class_colors() -> dict[str, tuple]:
    """Returns {specific_class_name: RGBA color}, shaded within each
    coordinate group's hue family. Shades are sampled from 0.45-0.95 of
    the colormap (skipping the near-white low end, which is hard to see
    against a white plot background).
    """
    import matplotlib.pyplot as plt

    colors = {}
    for group, members in CURATED_TAXONOMY.items():
        cmap = plt.get_cmap(COORDINATE_CMAP_FAMILIES[group])
        n = len(members)
        shades = np.linspace(0.45, 0.95, n) if n > 1 else [0.75]
        for member, shade in zip(members, shades):
            colors[member] = cmap(shade)
    return colors


# 2.5% increments, 0% to 75% inclusive (31 levels) -- extended from the
# previous 0-70% range per explicit request to include the full trajectory
# up to 75%.
PRUNING_LEVELS_FOCUSED = [round(x, 3) for x in np.arange(0.00, 0.751, 0.025).tolist()]
