"""Materialise a dataset's images single-process, before any torchrun sweep.

VLMEvalKit decodes images out of the benchmark TSV lazily, inside
``dump_image``, guarded only by ``read_ok(path)``. Under torchrun every rank
races to write the same file, so a rank can open a half-written JPEG and raise
``PIL.UnidentifiedImageError``. That kills one rank; the others then block in a
collective until NCCL times out -- measured at 68 minutes on DocVQA_VAL, for a
file that was perfectly valid seconds later.

Running this first makes the first sweep of a new dataset cost one extra
single-process pass and removes the race entirely.
"""

import sys

from vlmeval.dataset import build_dataset


def main(name):
    ds = build_dataset(name)
    if ds is None:
        print(f"{name}: build_dataset returned None", flush=True)
        return 1
    n = len(ds.data)
    print(f"{name}: extracting images for {n} rows", flush=True)
    bad = 0
    for i in range(n):
        try:
            ds.dump_image(ds.data.iloc[i])
        except Exception as e:  # noqa: BLE001 - report and continue, one bad row
            bad += 1                                  # must not sink the sweep
            if bad <= 5:
                print(f"  row {i}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    print(f"{name}: done, {bad} rows failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
