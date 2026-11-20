"""Stage only the LLP split needed for an event-error audit.

No video downloads, model training, or 41k-file full extraction. It uses
saved Drive annotations and saved original `feats.zip`. Read-only to Drive;
extracts just the requested split's three pretrained .npy files to /content.
"""
from __future__ import annotations
import argparse
import csv
import shutil
from pathlib import Path
from zipfile import ZipFile


def split_ids(manifest: Path):
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing split annotation CSV: {manifest}')
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        header = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter='\t' if '\t' in header else ',')
        if 'filename' not in (reader.fieldnames or []):
            raise ValueError(f'No filename column: {manifest}')
        ids = [row['filename'].strip()[:11] for row in reader if row.get('filename')]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate normalized identifiers in {manifest}')
    return ids


def prepare(artifact_root: Path, project_root: Path, split: str):
    manifest_name = 'AVVP_test_pd.csv' if split == 'test' else 'AVVP_val_pd.csv'
    original_ann = artifact_root / 'data/raw/llp'
    local_ann = project_root / 'data/raw/llp'
    local_feats = project_root / 'data/processed/feats'
    if not (original_ann/manifest_name).is_file() and not (local_ann/manifest_name).is_file():
        raise FileNotFoundError(f'Cannot find {manifest_name} in {original_ann} or {local_ann}')
    local_ann.mkdir(parents=True,exist_ok=True)
    for name in [manifest_name,'AVVP_eval_audio.csv','AVVP_eval_visual.csv']:
        src=original_ann/name
        dst=local_ann/name
        if not dst.exists():
            if not src.is_file():raise FileNotFoundError(f'Missing annotation file: {src}')
            shutil.copyfile(src,dst)
    ids=split_ids(local_ann/manifest_name)
    expected_n = 1200 if split=='test' else 649
    if len(ids) != expected_n:
        raise ValueError(f'Expected {expected_n} clips; got {len(ids)}')
    modalities=['vggish','res152','r2plus1d_18']
    needed=[(m,video_id) for video_id in ids for m in modalities
            if not (local_feats/m/(video_id+'.npy')).is_file()]
    print(f'{split.upper()} FILES: {len(ids)} clips; {len(needed)} feature arrays not yet extracted',flush=True)
    if not needed:
        print('ERROR AUDIT INPUTS READY',flush=True)
        return local_ann,local_feats
    archive=artifact_root/'data/feats.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'Original feature archive required: {archive}')
    with ZipFile(archive,'r') as z:
        mapping={}
        for n in z.namelist():
            parts=n.replace('\\','/').split('/')
            if len(parts)<2 or parts[-2] not in modalities or not parts[-1].endswith('.npy'):
                continue
            key=(parts[-2],parts[-1][:-4])
            if key in mapping:
                raise ValueError(f'Duplicate feature in ZIP: {key}')
            mapping[key]=n
        missing=[f'{m}/{id}.npy' for m,id in needed if (m,id) not in mapping]
        if missing:
            raise FileNotFoundError('Some features not in archive, e.g.: '+', '.join(missing[:8]))
        for i,(m,video_id) in enumerate(needed,1):
            dest=local_feats/m/(video_id+'.npy')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(mapping[(m,video_id)]) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            if i%450 == 0 or i == len(needed):
                print(f'Extracted {i}/{len(needed)} saved feature arrays',flush=True)
    print('ERROR AUDIT INPUTS READY',flush=True)
    return local_ann,local_feats


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--split',choices=['test','validation'],default='test')
    args=parser.parse_args()
    prepare(args.artifact_root.expanduser().resolve(),args.project_root.expanduser().resolve(),args.split)
