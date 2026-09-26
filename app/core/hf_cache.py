"""Keep downloaded models inside the project instead of ~/.cache/huggingface.

HF_HOME is read by huggingface_hub when it is first imported, so this module
must be imported before transformers / open_clip / huggingface_hub. Importing
it for its side effect is the point — there is nothing to call.

Override by exporting HF_HOME yourself; the default only applies if unset.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_HOME = PROJECT_ROOT / "data" / "hf"

os.environ.setdefault("HF_HOME", str(HF_HOME))

# A slow link needs more than the 10s default, or large weight files time out.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
