"""Render the verified Fig. 4 panels side by side in one IEEE column."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "figures"

def main():
    curves = pd.read_csv(ROOT / "tables/referral_curves_by_seed.csv")
    summary = pd.read_csv(ROOT / "tables/fusion_weights_mean_sd_ci.csv")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 6.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.major.pad": 2, "ytick.major.pad": 2})
    figure = plt.figure(figsize=(3.5, 2.6))
    axes = [figure.add_axes([.18, .435, .30, .41]),
            figure.add_axes([.65, .435, .32, .41])]
    methods = ["Standard JointEnergy", "Normal-Aware JointEnergy", "MC Dropout", "SafeFundus fusion"]
    labels = ["Standard JointEnergy", "Normal-Aware JointEnergy", "MC Dropout", "Fixed 0.5/0.5 fusion"]
    handles = []
    for method, color, label, style in zip(methods, ["#777777", "#0072B2", "#D55E00", "#009E73"], labels, [":", "--", "-.", "-"]):
        data = curves[curves.method == method].groupby("test_familiar_burden").unknown_referral.mean()
        mask = data.index <= .25
        x, y = data.index[mask].to_numpy() * 100, data.values[mask] * 100
        line, = axes[0].plot(x, y, color=color, label=label, ls=style, lw=1.1, drawstyle="steps-post")
        np.testing.assert_array_equal(line.get_ydata(), y)
        handles.append(line)
    for budget in [5, 10, 15]:
        axes[0].axvline(budget, color="#BBBBBB", lw=.45, zorder=0)
    axes[0].set(xlim=(0, 25), ylim=(0, 100), xticks=[0, 5, 10, 15, 20, 25],
                xlabel="Familiar test referral\nburden (%)",
                ylabel="Unknown referral\nsensitivity (%)", title="(a) Matched burden\n(retrospective)")
    handles_b = []
    labels_b = []
    for metric, color, label, marker in [("unknown_referral", "#0072B2", "Unknown referral", "o"),
                                        ("r2n", "#D55E00", "Rare-to-Normal Error", "s"),
                                        ("familiar_referral", "#777777", "Familiar referral", "^")]:
        data = summary[summary.metric == metric].sort_values("alpha_energy")
        line, = axes[1].plot(data.alpha_energy, data["mean"] * 100, color=color,
                             marker=marker, label=label, lw=1.05, ms=2.6)
        np.testing.assert_array_equal(line.get_ydata(), data["mean"].to_numpy() * 100)
        handles_b.append(line)
        labels_b.append(label)
    axes[1].axvline(.5, color="#555555", ls=":", lw=.8)
    axes[1].text(.53, 95, "Original\nweight", va="top", fontsize=6.2, color="#555555")
    axes[1].set(xlim=(-.05, 1.05), ylim=(0, 100), xticks=[0, .25, .5, .75, 1],
                xlabel="Energy weight $\\alpha$\n(MC: $1-\\alpha$)", ylabel="Rate (%)",
                title="(b) Validation-calibrated\nweight sweep")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", lw=.4)
        axis.tick_params(length=2.4, width=.6)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(.015, .255),
                  fontsize=6.4, frameon=False, borderaxespad=0,
                  handlelength=1.6, labelspacing=.25, handletextpad=.5)
    figure.legend(handles_b, labels_b, loc="upper left", bbox_to_anchor=(.56, .255),
                  fontsize=6.4, frameon=False, borderaxespad=0,
                  handlelength=1.6, labelspacing=.25, handletextpad=.5)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    page = figure.bbox
    for label in figure.findobj(match=matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            assert bounds.x0 >= page.x0 - .1 and bounds.x1 <= page.x1 + .1, label.get_text()
            assert bounds.y0 >= page.y0 - .1 and bounds.y1 <= page.y1 + .1, label.get_text()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metadata = {"Author": "", "Creator": "Matplotlib", "Title": "SafeFundus reviewer analyses"}
    figure.savefig(DESTINATION / "reviewer_analyses_compact_row.pdf", metadata=metadata)
    figure.savefig(DESTINATION / "reviewer_analyses_compact_row.png", dpi=400)
    plt.close(figure)
    print("PASS: unchanged plotted data; one-column 3.5 x 2.6 inch horizontal layout; no clipped labels.")

if __name__ == "__main__":
    main()
