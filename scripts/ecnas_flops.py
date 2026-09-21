"""Recompute EC-NAS architecture parameters and FLOPs from the cell specification.

Reimplements NAS-Bench-101's cell construction (channel assignment, per-vertex
ops, input/output projections) in pure Python, so that EC-NAS's convolutional
architectures can be described by their *actual* computational cost rather than
by a transform of their parameter count.

The reconstruction is only trusted if it reproduces the `trainable_params` that
EC-NAS itself recorded for every architecture. Run:

    python scripts/ecnas_flops.py validate

to grid-search the macro skeleton against the recorded parameter counts.

Structure follows nasbench/lib/model_builder.py (google-research/nasbench):
  - stem:            conv3x3 -> BN -> ReLU, `stem_filters` output channels
  - stacks:          `num_stacks` stacks of `modules_per_stack` cells
  - channels:        doubled at the start of each stack after the first
  - each cell:       channel assignment via compute_vertex_channels, one op per
                     interior vertex, 1x1 projection for input->vertex edges and
                     for the input->output edge
  - classifier:      global average pool -> dense(num_labels)

Operation codes, as decoded from the architectures' own graph files:
  -1 input, -2 output, 0 conv3x3-bn-relu, 1 conv1x1-bn-relu, 2 maxpool3x3

Parameter counts per layer, matching base_ops.conv_bn_relu (conv with
use_bias=False, then batch normalisation with gamma and beta):
  conv kxk : k*k*C_in*C_out + 2*C_out
  maxpool  : 0
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from dataclasses import dataclass, replace

CONV3X3, CONV1X1, MAXPOOL = 0, 1, 2
INPUT, OUTPUT = -1, -2

GRAPHS_5V9E = "data/raw/ec_nas/graphs/generated_graphs_5V9E.json"
FEATURES = "data/processed/ec_nas/ec_nas_features.csv"


@dataclass(frozen=True)
class Skeleton:
    """Macro body the cell is inserted into.

    The canonical NAS-Bench-101 configuration is stem 128, three stacks of
    three cells with the channel count doubling per stack, and a biased
    classifier. The grid search overrides these, because EC-NAS's own recorded
    parameter counts do not match the canonical configuration.
    """
    stem_filters: int = 128
    num_stacks: int = 3
    modules_per_stack: int = 3
    num_labels: int = 10
    bn: bool = True
    double_channels: bool = True
    classifier_bias: bool = True

    def stack_channels(self) -> list[int]:
        chans, out = self.stem_filters, []
        for stack in range(self.num_stacks):
            if stack > 0 and self.double_channels:
                chans *= 2
            out.append(chans)
        return out


def compute_vertex_channels(input_channels: int, output_channels: int,
                            matrix: list[list[int]]) -> list[int]:
    """Channel count at every vertex, per NAS-Bench-101's algorithm."""
    n = len(matrix)
    vc = [0] * n
    vc[0] = input_channels
    vc[n - 1] = output_channels
    if n == 2:
        return vc

    in_degree = [sum(matrix[src][dst] for src in range(1, n)) for dst in range(n)]
    interior = output_channels // in_degree[n - 1]
    correction = output_channels % in_degree[n - 1]

    for v in range(1, n - 1):
        if matrix[v][n - 1]:
            vc[v] = interior
            if correction:
                vc[v] += 1
                correction -= 1

    for v in range(n - 3, 0, -1):
        if not matrix[v][n - 1]:
            for dst in range(v + 1, n - 1):
                if matrix[v][dst]:
                    vc[v] = max(vc[v], vc[dst])

    final_fan_in = sum(vc[v] for v in range(1, n - 1) if matrix[v][n - 1])
    assert final_fan_in == output_channels or n == 2, "channel assignment inconsistent"
    return vc


def conv_params(kernel: int, c_in: int, c_out: int, bn: bool) -> int:
    return kernel * kernel * c_in * c_out + (2 * c_out if bn else 0)


def cell_params(matrix: list[list[int]], ops: list[int], channels: int,
                skeleton: Skeleton) -> tuple[int, int]:
    """Trainable parameters and multiply-accumulates for one cell.

    Returns (params, macs) where macs counts multiply-accumulate operations, so
    FLOPs = 2 * macs * spatial_positions. The spatial extent is applied by the
    caller, since it depends on which stack the cell sits in.
    """
    n = len(matrix)
    vc = compute_vertex_channels(channels, channels, matrix)
    params = macs = 0

    for t in range(1, n - 1):
        # 1x1 projection of the cell input onto this vertex, if directly wired.
        if matrix[0][t]:
            params += conv_params(1, channels, vc[t], skeleton.bn)
            macs += 1 * channels * vc[t]

        op = ops[t]
        if op == CONV3X3:
            params += conv_params(3, vc[t], vc[t], skeleton.bn)
            macs += 9 * vc[t] * vc[t]
        elif op == CONV1X1:
            params += conv_params(1, vc[t], vc[t], skeleton.bn)
            macs += 1 * vc[t] * vc[t]
        elif op == MAXPOOL:
            pass
        else:
            raise ValueError(f"unexpected op code {op}")

    has_fan_in = any(matrix[t][n - 1] for t in range(1, n - 1))
    if not has_fan_in:
        params += conv_params(1, channels, channels, skeleton.bn)
        macs += 1 * channels * channels
    elif matrix[0][n - 1]:
        params += conv_params(1, channels, channels, skeleton.bn)
        macs += 1 * channels * channels

    return params, macs


def model_params_and_macs(matrix, ops, skeleton: Skeleton,
                          spatial: tuple[int, ...]) -> tuple[int, int]:
    """Total trainable parameters and MACs for the full network.

    `spatial` gives the spatial extent (H = W) of each stack's feature map.
    """
    params = conv_params(3, 3, skeleton.stem_filters, skeleton.bn)
    macs = 9 * 3 * skeleton.stem_filters * (spatial[0] ** 2)

    for stack, channels in enumerate(skeleton.stack_channels()):
        hw = spatial[stack]
        for _ in range(skeleton.modules_per_stack):
            cp, cm = cell_params(matrix, ops, channels, skeleton)
            params += cp
            macs += cm * hw * hw

    final_channels = skeleton.stem_filters * (
        2 ** (skeleton.num_stacks - 1) if skeleton.double_channels else 1
    )
    params += final_channels * skeleton.num_labels
    if skeleton.classifier_bias:
        params += skeleton.num_labels
    return params, macs


def load_ecnas() -> list[tuple[str, list[list[int]], list[int], int]]:
    """(hash, adjacency, ops, recorded trainable_params) for the 5V9E space."""
    graphs = json.load(open(GRAPHS_5V9E, encoding="utf-8"))
    recorded: dict[str, int] = {}
    with open(FEATURES, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["run_id"].startswith("5V9E_"):
                recorded[row["run_id"].split("_")[1]] = int(row["params"])
    out = []
    for h, (matrix, ops) in graphs.items():
        if h in recorded:
            out.append((h, matrix, ops, recorded[h]))
    return out


def validate(skeleton: Skeleton, verbose: bool = False) -> float:
    """Fraction of architectures whose recorded parameter count is reproduced."""
    archs = load_ecnas()
    spatial = tuple(32 // (2 ** i) for i in range(skeleton.num_stacks))
    ok = 0
    examples = []
    for h, matrix, ops, rec in archs:
        pred, _ = model_params_and_macs(matrix, ops, skeleton, spatial)
        if pred == rec:
            ok += 1
        elif verbose and len(examples) < 5:
            examples.append((h, len(matrix), ops, rec, pred))
    if verbose and examples:
        for h, n, ops, rec, pred in examples:
            print(f"    mismatch |V|={n} ops={ops} recorded={rec} predicted={pred}")
    return ok / len(archs)


def widths_matching(matrix, ops, recorded: int, skeleton: Skeleton,
                    limit: int = 4096) -> list[int]:
    """Every channel width for which this architecture's parameters are reproduced."""
    spatial = tuple(32 // (2 ** i) for i in range(skeleton.num_stacks))
    hits = []
    for width in range(1, limit + 1):
        sk = replace(skeleton, stem_filters=width)
        try:
            params, _ = model_params_and_macs(matrix, ops, sk, spatial)
        except (AssertionError, ValueError, ZeroDivisionError):
            continue
        if params == recorded:
            hits.append(width)
    return hits


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"
    if mode not in ("validate", "solve"):
        print(__doc__)
        return

    archs = load_ecnas()
    print(f"architectures with recorded params: {len(archs)}")
    print(f"recorded params range: {min(a[3] for a in archs):,} - {max(a[3] for a in archs):,}")
    print()

    if mode == "validate":
        print("=== canonical NAS-Bench-101 skeleton ===")
        canon = Skeleton()
        print(f"  stacks={canon.num_stacks} modules={canon.modules_per_stack} "
              f"stem={canon.stem_filters} match = {validate(canon, verbose=True):.1%}")
        print()

        print("=== grid search over macro bodies ===")
        best = []
        for stacks, modules, stem, bn, dbl, bias in itertools.product(
            (1, 2, 3, 4), (1, 2, 3, 4), (16, 32, 64, 128, 256, 512),
            (True, False), (True, False), (True, False)
        ):
            sk = Skeleton(stem_filters=stem, num_stacks=stacks,
                          modules_per_stack=modules, bn=bn,
                          double_channels=dbl, classifier_bias=bias)
            try:
                frac = validate(sk)
            except (AssertionError, ValueError, ZeroDivisionError):
                continue
            if frac > 0.0:
                best.append((frac, sk))
        best.sort(key=lambda t: -t[0])
        if not best:
            print("  no macro body in the search space reproduced any architecture")
        for frac, sk in best[:10]:
            print(f"  {frac:6.1%}  stacks={sk.num_stacks} modules={sk.modules_per_stack} "
                  f"stem={sk.stem_filters} bn={sk.bn} double={sk.double_channels} "
                  f"clf_bias={sk.classifier_bias}")
        return

    # "solve": for a handful of architectures spanning the space, find every
    # channel width that reproduces the recorded parameter count. If one macro
    # body is right, the same width should recur across architectures.
    print("=== solving for channel width per architecture ===")
    by_key: dict[tuple, list] = {}
    for h, matrix, ops, rec in archs:
        by_key.setdefault((len(matrix), tuple(ops)), []).append((h, matrix, ops, rec))

    candidates = [
        Skeleton(),                                              # canonical
        Skeleton(double_channels=False),
        Skeleton(double_channels=False, classifier_bias=False),
        Skeleton(classifier_bias=False),
        Skeleton(num_stacks=3, modules_per_stack=3, double_channels=False, bn=False),
    ]
    # one representative architecture from each of the 20 (|V|, ops) groups
    sample = [sorted(v)[len(v) // 2] for v in by_key.values()][:8]

    for idx, sk in enumerate(candidates):
        label = (f"stacks={sk.num_stacks} modules={sk.modules_per_stack} "
                 f"double={sk.double_channels} bn={sk.bn} clf_bias={sk.classifier_bias}")
        print(f"\n--- candidate {idx}: {label}")
        for h, matrix, ops, rec in sample:
            hits = widths_matching(matrix, ops, rec, sk)
            print(f"    |V|={len(matrix)} ops={ops} rec={rec:>10,} -> widths {hits[:6]}"
                  f"{' ...' if len(hits) > 6 else ''}")


if __name__ == "__main__":
    main()
