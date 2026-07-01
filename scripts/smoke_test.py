"""One-seed smoke test of the full pipeline (run before the full matrix)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all_seeds import run_model

SEED = 42
for m in ["vae", "ctgan", "tvae", "gc"]:
    r = run_model(m, SEED)
    print(f">>> {m}: fid={r['fidelity_score']:.3f} util={r['utility_ratio']:.3f} "
          f"dcr_ratio={r['dcr_ratio']:.3f} mia_auc={r['mia_auc_domias']:.3f} "
          f"mia_clf={r['mia_auc_domias_clf']:.3f} c2st={r['c2st_acc']:.3f} "
          f"gap={r['dcr_mia_gap']} faa_pass={r['faa_pass']}")
