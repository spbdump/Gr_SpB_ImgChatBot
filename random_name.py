import random

def get_random_name():
    anim_names=["dog", "cat", "elephant", "tiger", "lion", 
                "giraffe", "zebra", "cheetah", "kangaroo", 
                "koala", "panda", "dolphin", "whale", 
                "penguin", "seal", "bear", "fox", "wolf", 
                "rabbit", "squirrel", "mouse", "rat", "hamster", 
                "guinea pig", "hedgehog", "raccoon", "skunk", 
                "bat", "otter", "chipmunk", "platypus", "crocodile", 
                "alligator", "rhinoceros", "hippopotamus", "gorilla", 
                "chimpanzee", "orangutan", "sloth", "lemur", "meerkat", 
                "mongoose", "armadillo", "anteater", "tapir", "capybara", 
                "beaver", "muskrat", "porcupine", "prairie dog", "bison", 
                "buffalo", "yak", "moose", "reindeer", "elk", "gazelle", 
                "antelope", "ibex", "marmoset", "tarsier", "leech", 
                "slug", "snail", "centipede", "millipede", "scorpion", 
                "tarantula", "octopus", "squid", "jellyfish", "seahorse", 
                "seagull", "pelican", "vulture", "condor", "heron", "egret", 
                "stork", "flamingo", "toucan", "woodpecker", "robin", "sparrow", 
                "crow", "falcon", "hawk", "bald_eagle", "owl", "parrot", 
                "cockatoo", "canary", "finch", "goldfish", "angelfish", 
                "clownfish", "seahorse", "starfish", "sea_urchin", 
                "sea_cucumber", "sea_lion", "sea_otter", "manatee", 
                "walrus", "narwhal", "beluga_whale", "sea_turtle", 
                "box turtle", "terrapin", "chameleon", "gecko", 
                "iguana", "komodo_dragon", "monitor lizard", "anaconda", 
                "boa constrictor", "python", "cobra", "rattlesnake", 
                "garter snake", "king snake", "newt", "salamander", 
                "axolotl", "tadpole", "bullfrog", "tree_frog", 
                "poison dart_frog", "parakeet", "budgerigar", "finch", 
                "canary", "lovebird", "hamster", "gerbil", "guinea_pig", 
                "rabbit", "ferret", "cockroach", "ladybug", "grasshopper", 
                "cricket", "praying mantis", "bumblebee", "honeybee", "wasp", 
                "hornet", "butterfly", "moth", "firefly", "dragonfly", 
                "damselfly", "mayfly", "cockroach", "termite", "cricket", 
                "grasshopper", "locust", "katydid", "bedbug", "flea", 
                "tick", "mosquito", "housefly", "horsefly", "barnacle", 
                "crab", "lobster", "shrimp", "prawn", "crayfish", 
                "sea_urchin", "sea_cucumber", "jellyfish", "coral", 
                "sponge", "starfish", "clam", "mussel", "oyster", 
                "snail", "slug", "earthworm", "millipede", "centipede", 
                "pill_bug", "scorpion", "tarantula", "spider", "praying_mantis", 
                "walking_stick", "leaf_insect", "stick_insect", "cockroach", 
                "beetle", "ladybug", "butterfly", "dragonfly", "firefly", 
                "moth", "cicada", "cricket", "grasshopper", "ant", "bee", 
                "wasp", "hornet", "termite", "mantis_shrimp", "narwhal", 
                "swordfish", "marlin", "sailfish", "barracuda", "mackerel", 
                "salmon", "trout", "bass", "catfish", "pike", "eel", 
                "seahorse", "clownfish", "damselfish", "tang", "lionfish", 
                "angelfish", "gobies", "blenny", "flounder", "halibut", 
                "sole", "moray_eel", "triggerfish", "parrotfish", "pufferfish", 
                "boxfish", "porcupinefish", "starfish", "sea_cucumber", 
                "sea_urchin", "jellyfish", "anemone", "coral", "sea_sponge", 
                "sea_slug", "nudibranch", "mantis_shrimp", "octopus", "squid", 
                "cuttlefish", "sea snail", "cone_snail", "cowrie", "abalone", 
                "scallop", "mussel", "oyster", "clam", "shrimp", "lobster", 
                "crab", "crayfish", "barnacle", "hermit crab", "horseshoe_crab", 
                "copepod", "krill", "plankton", "prawn", "squid", "octopus", 
                "cuttlefish", "jellyfish", "sea_anemone", "coral", "sponge", 
                "sea_urchin", "starfish", "sea_cucumber", "sand_dollar", 
                "sea_snail", "sea slug", "nudibranch", "mantis_shrimp", 
                "clam", "oyster", "mussel", "scallop", "crab", "lobster", 
                "shrimp", "barnacle", "hermit_crab", "horseshoe_crab", 
                "krill", "plankton", "prawn", "sea_dragon", "sea_horse", 
                "sea_lion", "sea_otter", "sea_turtle", "manatee", 
                "walrus", "narwhal", "beluga whale", "killer_whale", 
                "dolphin", "porpoise", "shark", "stingray", "manta_ray", 
                "anglerfish", "flashlight_fish", "gulper_eel", "viperfish", 
                "hatchetfish", "fangtooth", "dragonfish", "lanternfish", 
                "cookiecutter shark", "hammerhead_shark", "bull_shark", 
                "tiger_shark", "great_white_shark", "basking_shark", 
                "nurse_shark", "mako_shark", "sawfish", "paddlefish", 
                "sturgeon", "gar", "bowfin", ]
    names = []
    names.append(random.choice(anim_names))
    names.append(random.choice(anim_names))
    

    return '_'.join(names)