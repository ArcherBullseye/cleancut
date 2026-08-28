# Threshold defaults
DEFAULT_VISUAL_THRESHOLD: float = 0.7
DEFAULT_SCENE_THRESHOLD: float = 27.0
DEFAULT_VLM_GAPS_RADIUS: float = 30.0
DEFAULT_LLM_CONFIDENCE: float = 0.6
DEFAULT_VLM_CONFIDENCE: float = 0.55
DEFAULT_VLM_MIN_SHOT_DURATION: float = 0.4

# Text limits
MAX_REASON_LENGTH: int = 160

# Abort an LLM/VLM scan after this many consecutive per-item failures —
# a dead Ollama otherwise grinds through every item eating timeouts.
MAX_CONSECUTIVE_LLM_FAILURES: int = 5

# Categories surfaced by `review` by default. Violence cuts are hidden
# (fight scenes are kept by default); "multi" is an LLM combination label
# that always includes at least one focal category.
FOCAL_CATEGORIES: frozenset[str] = frozenset({"sex", "drugs", "nudity", "multi"})
