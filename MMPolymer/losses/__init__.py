# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("MMPolymer.losses." + file.name[:-3])
