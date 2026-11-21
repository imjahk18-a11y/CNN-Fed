"""Check content identity and tile separation for train, calibration and held-out patches."""
import hashlib
import json

import numpy as np
from datasets.compact_cache import atomic_json
from datasets.fullz import Scene
from experiments.distributed_ai import CACHE, SCENES
from experiments.phase4_pretrain import OUT


def main():
    folds = json.loads((OUT / "splits.json").read_text())
    scenes = {i: Scene(CACHE / s) for i, s in SCENES.items()}
    hashes = {}
    report = {}
    for f, roles in folds.items():

        def fingerprints(refs):
            values = set()
            for s, i in refs:
                if (s, i) not in hashes:
                    c, rgb, _ = scenes[s][i]
                    h = hashlib.sha256()
                    h.update(c.tobytes())
                    h.update(rgb.tobytes())
                    hashes[s, i] = h.digest()
                values.add(hashes[s, i])
            return values

        learn = fingerprints(
            roles["pretrain"] + roles["finetune"] + roles["client1"] + roles["client2"]
        )
        seen = {
            "calibration": fingerprints(roles["calibration"]),
            "validation": fingerprints(roles["validation"]),
            "audit": fingerprints(roles["audit"]),
        }
        dup = {f"learn_{role}": len(learn & values) for role, values in seen.items()}
        dup.update(
            val_audit=len(seen["validation"] & seen["audit"]),
            cal_val=len(seen["calibration"] & seen["validation"]),
            cal_audit=len(seen["calibration"] & seen["audit"]),
            client1_client2=len(fingerprints(roles["client1"]) & fingerprints(roles["client2"])),
        )

        def gap(a, b):
            sid = roles[a][0][0]
            assert sid == roles[b][0][0]
            xy = scenes[sid].tile_xy
            x = xy[[i for _, i in roles[a]]]
            y = xy[[i for _, i in roles[b]]]
            return int(np.abs(x[:, None, :] - y[None, :, :]).max(axis=2).min())

        report[f] = dict(
            duplicate_model_inputs=dup,
            min_calibration_pretrain_tile_distance=gap("calibration", "pretrain"),
            min_audit_validation_tile_distance=gap("audit", "validation"),
        )
        assert (
            not any(dup.values())
            and gap("audit", "validation") >= 3
            and gap("calibration", "pretrain") >= 3
        )
    atomic_json(OUT / "split_audit.json", report)


if __name__ == "__main__":
    main()
