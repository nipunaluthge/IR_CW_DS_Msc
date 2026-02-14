# src/config.py

CHROMA_PATH = "db_chroma"
ACTIONS_COLLECTION = "actions"
STRATEGIES_COLLECTION = "strategies"

TOP_K = 10

# Distance -> similarity conversion:
# Chroma returns distances (lower is better). We'll convert to similarity in [0,1].
# similarity = 1 / (1 + distance)
def dist_to_sim(d: float) -> float:
    return 1.0 / (1.0 + float(d))

# Thresholds for labels
HIGH_SIM = 0.55
MED_SIM  = 0.40
