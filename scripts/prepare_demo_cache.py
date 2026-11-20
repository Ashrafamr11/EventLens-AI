"""Package a PREVIOUSLY EXTRACTED, real-video .npz for a five-minute live demo.

This is explicitly cached-feature inference, not new-video end-to-end extraction.
Does not use a remote AI API or compute synthetic fake feature vectors.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

SHAPES={'audio':(10,128),'visual':(10,2048),'motion':(10,512)}

def validate_and_copy(source:Path,dest:Path):
    with np.load(source,allow_pickle=False) as z:
        if set(SHAPES)-set(z.files):
            raise ValueError('NPZ must contain audio, visual and motion from a real source video')
        arrays={k:np.asarray(z[k],dtype=np.float32) for k in SHAPES}
    for name,shape in SHAPES.items():
        if arrays[name].shape!=shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f'{name}: expected finite {shape}, got {arrays[name].shape}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    np.savez_compressed(dest,**arrays)
    print(f'VALID CACHED FEATURE DEMO: {dest}; upload via Features (.npz) mode')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path)
    args=p.parse_args()
    validate_and_copy(args.source,args.destination)
