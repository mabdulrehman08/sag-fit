"""Benchmark clearance from synthetic 3D conductor and vegetation points."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


HALF_SPAN = 50.0
GRID = np.linspace(-HALF_SPAN, HALF_SPAN, 2001)
DEV_SEEDS = range(8)
HELDOUT_SEEDS = range(1000, 1040)
METHODS = ("raw", "quadratic", "catenary")
RANSAC_DRAWS = 48
INLIER_RADIUS = 0.85
SWAY_PHASES = 64


@dataclass(frozen=True)
class Scene:
    name: str
    noise_m: float
    gap_m: int
    outlier_fraction: float = 0.12
    partial: bool = False
    sway_m: float = 0.0
    lateral_bow_m: float = 0.0


SCENES = (
    Scene("Dense clean", 0.0, 0, outlier_fraction=0.0),
    *(Scene(f"Noise {noise:.2f}, gap {gap}", noise, gap)
      for noise in (0.06, 0.20, 0.45) for gap in (0, 12, 28)),
    Scene("Partial view", 0.20, 0, partial=True),
    Scene("Lateral bow", 0.20, 12, lateral_bow_m=1.2),
    Scene("Sway", 0.20, 12, sway_m=1.4),
)


def catenary(u, scale, center, bottom):
    # u, scale and center are expressed in half-span units.
    return bottom + 2 * HALF_SPAN * scale * np.sinh((u - center) / (2 * scale)) ** 2


def simulate(scene, seed):
    """Return observed wire, vegetation, and the ideal clearance in meters."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(1.5, 5.2)
    center = rng.uniform(-0.18, 0.18)
    bottom = rng.uniform(9, 13)
    slope = rng.uniform(-0.12, 0.12)
    offset = rng.uniform(-2, 2)

    def still_wire(x):
        u = x / HALF_SPAN
        y = offset + slope * x
        z = catenary(u, scale, center, bottom)
        return np.column_stack((x, y, z))

    def wire(x, phase):
        points = still_wire(x)
        envelope = np.cos(np.pi * x / (2 * HALF_SPAN))
        points[:, 1] += (scene.lateral_bow_m + scene.sway_m * np.sin(phase)) * envelope
        points[:, 2] += 0.30 * scene.sway_m * np.cos(phase) * envelope
        return points

    # Generate two fixed clusters of branch tips relative to the still wire.
    # The first is near midspan, where the central gap can hide the wire.
    vegetation = []
    for tip_x, side in ((rng.uniform(-6, 6), 1), (rng.uniform(15, 36), -1)):
        tip = still_wire(np.array([tip_x]))[0]
        tip += [0, side * rng.uniform(1.5, 3.0), -rng.uniform(0.45, 1.6)]
        depth = rng.uniform(0, 1, 100)
        branch = rng.normal(size=(100, 3)) * [1.8, 0.17, 0.15]
        branch += depth[:, None] * [3.0, side * 1.1, -2.0]
        vegetation.append(tip + branch)
    vegetation = np.vstack(vegetation)
    tree = cKDTree(vegetation)

    # Correlate capture phase with chainage to mimic sequential observations.
    n = 360 if scene.noise_m == 0 else 180
    x = np.sort(rng.uniform(-HALF_SPAN, HALF_SPAN, n))
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * x / (2 * HALF_SPAN)
    observed = wire(x, phase)
    observed += scene.noise_m * rng.normal(size=(n, 3)) * [0.6, 1.5, 1.0]
    outliers = rng.choice(n, round(n * scene.outlier_fraction), replace=False)
    observed[outliers, 1:] += rng.normal(0, 2.8, size=(len(outliers), 2))
    visible = np.abs(observed[:, 0]) <= HALF_SPAN
    if scene.gap_m:
        visible &= np.abs(x) >= scene.gap_m / 2
    if scene.partial:
        visible &= (x >= -44) & (x <= -8)

    phases = np.linspace(0, 2 * np.pi, SWAY_PHASES, endpoint=False) if scene.sway_m else [0]
    truth = min(clearance(wire(GRID, phase), tree) for phase in phases)
    return observed[visible], tree, truth


def clearance(wire, vegetation_tree):
    """Minimum 3D Euclidean distance between sampled wire and vegetation."""
    return float(vegetation_tree.query(wire)[0].min())


def height(params, u, method):
    if method == "quadratic":
        return np.polynomial.polynomial.polyval(u, params)
    return catenary(u, *params)


CAT_BOUNDS = ((0.9, -1.5, -np.inf), (22.0, 1.5, np.inf))


def catenary_jac(params, u):
    scale, center, _ = params
    v = (u - center) / scale
    return np.column_stack((
        HALF_SPAN * (2 * np.sinh(v / 2) ** 2 - v * np.sinh(v)),
        -HALF_SPAN * np.sinh(v),
        np.ones_like(u),
    ))


def fit_wire(points, method, seed):
    """Fit a lateral line and a vertical model using shared 3D observations."""
    u = points[:, 0] / HALF_SPAN
    yz = points[:, 1:]
    lateral = np.column_stack((np.ones_like(u), u))
    quadratic = np.column_stack((lateral, u ** 2))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(RANSAC_DRAWS):
        idx = rng.choice(len(points), 4, replace=False)
        sample_u = u[idx]
        if np.ptp(sample_u) < 0.25:
            continue
        y_params = np.linalg.lstsq(lateral[idx], yz[idx, 0], rcond=None)[0]
        z_params = np.linalg.lstsq(quadratic[idx], yz[idx, 1], rcond=None)[0]
        if method == "catenary":
            intercept, linear, curvature = z_params
            center = np.clip(-linear / (2 * max(curvature, 1e-6)), -1.5, 1.5)
            scale = np.clip(HALF_SPAN / (2 * max(curvature, 1e-6)), 0.9, 22.0)
            start = [scale, center, intercept - catenary(0, scale, center, 0)]
            sample_z = yz[idx, 1]
            z_params = least_squares(
                lambda p: catenary(sample_u, *p) - sample_z,
                start, jac=lambda p: catenary_jac(p, sample_u),
                bounds=CAT_BOUNDS, max_nfev=18,
            ).x
        predicted = np.column_stack((lateral @ y_params, height(z_params, u, method)))
        distances = np.linalg.norm(predicted - yz, axis=1)
        inliers = distances < INLIER_RADIUS
        if inliers.sum() < 4:
            continue
        score = (int(inliers.sum()), -float(np.square(distances[inliers]).sum()))
        if best is None or score > best[0]:
            best = score, inliers, y_params, z_params

    if best is None:
        return None
    _, inliers, y_params, z_params = best
    inlier_u = u[inliers]
    inlier_lateral = lateral[inliers]
    inlier_y, inlier_z = yz[inliers].T
    y_fit = least_squares(
        lambda p: inlier_lateral @ p - inlier_y, y_params,
        jac=lambda p: inlier_lateral.copy(), loss="soft_l1", f_scale=INLIER_RADIUS,
    )
    z_fit = least_squares(
        lambda p: height(p, inlier_u, method) - inlier_z, z_params,
        jac=(lambda p: catenary_jac(p, inlier_u)) if method == "catenary"
        else (lambda p: quadratic[inliers]),
        bounds=CAT_BOUNDS if method == "catenary" else (-np.inf, np.inf),
        loss="soft_l1", f_scale=INLIER_RADIUS,
    )
    if not (y_fit.success and z_fit.success):
        return None
    grid_u = GRID / HALF_SPAN
    return np.column_stack((
        GRID, y_fit.x[0] + y_fit.x[1] * grid_u, height(z_fit.x, grid_u, method),
    ))


def run_trial(scene, seed):
    observed, vegetation, truth = simulate(scene, seed)
    estimates = {"raw": clearance(observed, vegetation)}
    for method in METHODS[1:]:
        fitted = fit_wire(observed, method, seed + 10_000)
        estimates[method] = clearance(fitted, vegetation) if fitted is not None else None
    return {
        "seed": seed, "wire_points": len(observed), "truth_m": truth,
        "estimate_m": estimates,
        "signed_error_m": {
            method: value - truth if value is not None else None
            for method, value in estimates.items()
        },
    }


def summarize(trials):
    summary = {}
    for method in METHODS:
        errors = [trial["signed_error_m"][method] for trial in trials]
        valid = np.array([error for error in errors if error is not None])
        absolute = np.abs(valid)
        summary[method] = {
            "mae_m": float(absolute.mean()) if valid.size else None,
            "p90_absolute_m": float(np.quantile(absolute, 0.90)) if valid.size else None,
            "bias_m": float(valid.mean()) if valid.size else None,
            "overestimates": int(np.count_nonzero(valid > 0)),
            "failures": len(trials) - len(valid),
        }
    return summary


def benchmark(split):
    seeds = list(DEV_SEEDS if split == "dev" else HELDOUT_SEEDS)
    report = {
        "split": split, "seeds": seeds,
        "settings": {
            "span_m": 2 * HALF_SPAN, "grid_step_m": float(GRID[1] - GRID[0]),
            "ransac_draws": RANSAC_DRAWS, "inlier_radius_m": INLIER_RADIUS,
            "sway_truth_phases": SWAY_PHASES,
        },
        "scenes": [],
    }
    for scene in SCENES:
        trials = [run_trial(scene, seed) for seed in seeds]
        report["scenes"].append({
            "scene": asdict(scene), "summary": summarize(trials), "trials": trials,
        })
    return report


def print_table(report):
    print(f'{report["split"]}: {len(report["seeds"])} seeds per scene')
    print("Each method: MAE / P90 absolute error / signed bias (m), failures")
    print(f'{"Scene":<22} {"Raw points":>27} {"Quadratic":>27} {"Catenary":>27}')
    for row in report["scenes"]:
        cells = []
        for method in METHODS:
            stats = row["summary"][method]
            values = [stats[key] for key in ("mae_m", "p90_absolute_m", "bias_m")]
            metrics = " / ".join("n/a" if v is None else f"{v:.3f}" for v in values)
            cells.append(f'{metrics}, {stats["failures"]}')
        print(f'{row["scene"]["name"]:<22} ' + " ".join(f"{cell:>27}" for cell in cells))


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["scenes"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.3), sharey=True)
    fig.suptitle("Wire-to-vegetation clearance error", x=0.225, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(
        0.225, 0.909,
        f'{len(report["seeds"])} {report["split"]} seeds per scene | Log scale beyond +/-0.1 m',
        fontsize=10, color="#475569",
    )
    for ax, metric, label in zip(
        axes, ("mae_m", "bias_m"), ("Mean absolute error (m)", "Mean signed error (m)"),
    ):
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.23)
        ax.axvline(0, color="#94a3b8", linewidth=0.9)
        for method, name, color, marker, offset in zip(
            METHODS, ("Raw points", "Quadratic", "Catenary"),
            ("#475569", "#087f8c", "#bb511a"), ("o", "s", "^"), (-0.23, 0, 0.23),
        ):
            values = np.array([row["summary"][method][metric] for row in rows], dtype=float)
            ax.scatter(values, y + offset, label=name, color=color, marker=marker, s=29)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(label)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.axhspan(len(rows) - 1.5, len(rows) - 0.5, color="#f1f5f9", zorder=0)
    axes[0].set_yticks(y, [row["scene"]["name"] for row in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].set_xlim(left=-0.006)
    tick_options = (
        (0, 0.05, 0.1, 0.5, 1, 2, 5),
        (-5, -2, -1, -0.1, 0, 0.1, 0.5, 1, 2, 5),
    )
    for ax, options in zip(axes, tick_options):
        left, right = ax.get_xlim()
        ticks = [tick for tick in options if left <= tick <= right]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.225, right=0.975, top=0.87, bottom=0.14, wspace=0.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    report = benchmark(args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "clearance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii",
    )
    plot(report, args.out_dir / "clearance.png")
    print_table(report)
    print(f"Wrote clearance.json and clearance.png to {args.out_dir}")


if __name__ == "__main__":
    main()
