"""Dataset preparation entrypoint. Source archive must be acquired separately."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_training_assets import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=ROOT)
    args=p.parse_args()
    prepare(args.artifact_root.resolve(),args.project_root.resolve())
