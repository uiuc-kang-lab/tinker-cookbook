"""
Code grading bridge — imports SkyRL's evaluation logic to ensure exact
equivalence between the SkyRL and tinker-cookbook training pipelines.

The heavy lifting (subprocess isolation, reliability_guard, timeout
management) lives in ``examples.train.mix_code.code_eval``.

Set the SKYRL_PATH environment variable to point to the SkyRL checkout.
"""

import os
import sys
import dotenv
dotenv.load_dotenv()

_skyrl_path = os.environ.get("SKYRL_PATH")
if _skyrl_path is None:
    raise EnvironmentError(
        "SKYRL_PATH environment variable is not set. "
        "Point it to the root of the SkyRL repo, e.g. export SKYRL_PATH=/root/SkyRL"
    )
if _skyrl_path not in sys.path:
    sys.path.insert(0, _skyrl_path)

from examples.train.mix_code.code_eval import (  # noqa: E402, F401
    compute_score,
    extract_code_from_model,
)
