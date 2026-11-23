"""Reflow the verified Fig. 4 data vertically at true IEEE column width."""
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
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 9, "axes.titlesize": 10})
    figure = plt.figure(figsize=(3.5, 5.2))
    axes = [figure.add_axes([.18, .585, .785, .315]),
            figure.add_axes([.18, .095, .785, .315])]
    methods = ["Standard JointEnergy", "Normal-Aware JointEnergy", "MC Dropout", "SafeFundus fusion"]
    labels = ["Standard JointEnergy", "Normal-Aware JointEnergy", "MC Dropout", "Fixed 0.5/0.5 fusion"]
    for method, color, label, style in zip(methods, ["#777777", "#0072B2", "#D55E00", "#009E73"], labels, [":", "--", "-.", "-"]):
        data = curves[curves.method == method].groupby("test_familiar_burden").unknown_referral.mean()
        mask = data.index <= .25
        x, y = data.index[mask].to_numpy() * 100, data.values[mask] * 100
        line, = axes[0].plot(x, y, color=color, label=label, ls=style, lw=1.6, drawstyle="steps-post")
        np.testing.assert_array_equal(line.get_ydata(), y)
    for budget in [5, 10, 15]:
        axes[0].axvline(budget, color="#BBBBBB", lw=.6, zorder=0)
    axes[0].set(xlim=(0, 25), ylim=(0, 100), xlabel="Familiar test referral burden (%)",
                ylabel="Unknown referral sensitivity (%)", title="(a) Matched burden\n(retrospective)")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    for metric, color, label, marker in [("unknown_referral", "#0072B2", "Unknown referral", "o"),
                                        ("r2n", "#D55E00", "Rare-to-Normal Error", "s"),
                                        ("familiar_referral", "#777777", "Familiar referral", "^")]:
        data = summary[summary.metric == metric].sort_values("alpha_energy")
        line, = axes[1].plot(data.alpha_energy, data["mean"] * 100, color=color,
                             marker=marker, label=label, lw=1.5, ms=4)
        np.testing.assert_array_equal(line.get_ydata(), data["mean"].to_numpy() * 100)
    axes[1].axvline(.5, color="#555555", ls=":", lw=1)
    axes[1].text(.52, 95, "Original weight", fontsize=7, color="#555555")
    axes[1].set(xlim=(-.03, 1.03), ylim=(0, 100), xticks=[0, .25, .5, .75, 1],
                xlabel=r"Energy weight $\alpha$ (MC: $1-\alpha$)", ylabel="Rate (%)",
                title="(b) Validation-calibrated\nweight sweep")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, .88), fontsize=7, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", lw=.5)
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
    figure.savefig(DESTINATION / "reviewer_analyses_single_column.pdf", metadata=metadata)
    figure.savefig(DESTINATION / "reviewer_analyses_single_column.png", dpi=300)
    plt.close(figure)
    print("PASS: unchanged plotted data; original plot font sizes; all text inside 3.5 x 5.2 inch figure.")

if __name__ == "__main__":
    main()
