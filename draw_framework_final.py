"""
KuDA-Conflict final defense figure (clean 16:9 version)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, fc, ec, lw=2.0, r=0.15, z=3):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.03,rounding_size={r}",
        fc=fc,
        ec=ec,
        lw=lw,
        zorder=z,
    )
    ax.add_patch(p)


def text(ax, x, y, s, fs=12, c="#111827", bold=False, italic=False, ha="center", va="center", z=6):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color=c,
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",
        ha=ha,
        va=va,
        zorder=z,
    )


def arrow(ax, x1, y1, x2, y2, c="#334155", lw=2.2, style="-|>", rad=0.0, z=5):
    conn = f"arc3,rad={rad}" if abs(rad) > 1e-6 else "arc3,rad=0"
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=c, lw=lw, connectionstyle=conn),
        zorder=z,
    )


fig = plt.figure(figsize=(19.2, 10.8))
ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# palette
CV, CVL = "#2E7D32", "#E8F5E9"
CT, CTL = "#7B1FA2", "#F3E5F5"
CA, CAL = "#EF6C00", "#FFF3E0"
CY, CYL = "#F9A825", "#FFF8E1"
CP, CPL = "#D81B60", "#FCE4EC"
CIEC, CIECL = "#2E7D32", "#F1F8E9"
CICR, CICRL = "#EF6C00", "#FFF3E0"
CFUS, CFUSL = "#4527A0", "#EDE7F6"
COUT, COUTL = "#1565C0", "#E3F2FD"
CG = "#64748B"

yv, yt, ya = 72, 52, 32

# header
text(
    ax,
    50,
    96,
    "KuDA-Conflict: Final Framework (Defense Version)",
    fs=21,
    c="#0F172A",
    bold=True,
)
text(
    ax,
    50,
    92.5,
    "Backbone -> IEC (Token Compression) -> CSR Module + ICR -> C-Driven Dynamic Fusion -> Prediction",
    fs=12,
    c="#475569",
)

# section tags
tags = [
    (6, "Input"),
    (20, "Encoders"),
    (35, "Token-level Sentiment Projection"),
    (49, "IEC"),
    (64, "CSR Module + ICR"),
    (82, "Fusion"),
    (93, "Output"),
]
for x, t in tags:
    box(ax, x - 5, 85.2, 10, 4.8, "#F8FAFC", "#94A3B8", lw=1.3, r=0.1, z=2)
    text(ax, x, 87.6, t, fs=10.5, c="#334155", bold=True)

# input
for y, name, ec, fc, sub in [
    (yv, "Vision", CV, CVL, "frames"),
    (yt, "Text", CT, CTL, "tokens"),
    (ya, "Audio", CA, CAL, "frames"),
]:
    box(ax, 1.5, y - 4.6, 8.2, 9.2, fc, ec, lw=2.1, r=0.25)
    text(ax, 5.6, y + 1.1, name, fs=13, c=ec, bold=True)
    text(ax, 5.6, y - 1.9, sub, fs=9.5, c=ec, italic=True)

# encoders
for y, ec, fc, name, out in [
    (yv, CV, CVL, "FeatExt + Transformer", "H_V"),
    (yt, CT, CTL, "BERT Encoder", "H_T"),
    (ya, CA, CAL, "FeatExt + Transformer", "H_A"),
]:
    arrow(ax, 9.8, y, 12.2, y, c=ec, lw=2.0)
    box(ax, 12.2, y - 4.6, 15.2, 9.2, fc, ec, lw=2.0, r=0.25)
    text(ax, 19.8, y + 1.1, name, fs=10.8, c=ec, bold=True)
    text(ax, 19.8, y - 2.0, out, fs=11, c=ec, italic=True)

# sentiment projector
for y, ec in [(yv, CV), (yt, CT), (ya, CA)]:
    arrow(ax, 27.4, y, 30.2, y, c=ec, lw=2.0)
    box(ax, 30.2, y - 4.4, 10.2, 8.8, CYL, CY, lw=2.0, r=0.2)
    text(ax, 35.3, y + 1.1, "Token-level", fs=9.0, c="#A16207", bold=True)
    text(ax, 35.3, y - 0.8, "Sentiment Projection", fs=8.8, c="#A16207", bold=True)
    text(ax, 35.3, y - 2.8, "p_m  and  s_m = E[y|p_m]", fs=7.4, c="#A16207", italic=True)

# A module (shared pre-branch)
for y, ec in [(yv, CV), (yt, CT), (ya, CA)]:
    arrow(ax, 40.5, y, 43.0, y, c=ec, lw=2.0)
box(ax, 43.0, 24.0, 11.5, 56.0, CPL, CP, lw=2.5, r=0.3)
text(ax, 48.75, 75.0, "CSR Module", fs=15, c=CP, bold=True)
text(ax, 48.75, 70.7, "Cross-modal Sentiment Reference", fs=9.8, c=CP, bold=True)
text(ax, 48.75, 65.5, "V->T / A->T Cross-Attn", fs=9.3, c="#9D174D", italic=True)
text(ax, 48.75, 61.8, "outputs split to branches", fs=9.3, c="#9D174D", italic=True)

# IEC + ICR title (now sequential: IEC -> ICR)
box(ax, 56.0, 81.8, 23.0, 5.0, "#F8FAFC", "#94A3B8", lw=1.3, r=0.1, z=2)
text(ax, 67.5, 84.3, "IEC -> ICR (Sequential)", fs=11.0, c="#334155", bold=True)

# shared branch area
box(ax, 56.0, 24.0, 23.0, 56.0, "none", "#94A3B8", lw=1.2, r=0.2, z=1)
ax.plot([56.6, 78.4], [52.0, 52.0], color="#CBD5E1", lw=1.2, ls="--", zorder=1)

# top branch: IEC
box(ax, 57.2, 58.0, 20.6, 19.0, CIECL, CIEC, lw=2.2, r=0.22, z=4)
text(ax, 67.5, 73.0, "Branch-I: IEC (Information Extraction & Compression)", fs=10.4, c=CIEC, bold=True)
text(ax, 67.5, 68.8, "TextGuidedVisionPruner", fs=9.2, c=CIEC, bold=True)
text(ax, 67.5, 64.9, "Input: H_V (text-guided)", fs=10.2, c=CIEC, italic=True)
text(ax, 67.5, 61.6, "Top-K Prune + Merge  ->  H_V^IEC", fs=10.2, c=CIEC, italic=True)

# bottom branch: ICR
box(ax, 57.2, 27.0, 20.6, 22.5, CICRL, CICR, lw=2.2, r=0.22, z=4)
text(ax, 67.5, 45.8, "Branch-II: ICR (Inconsistency-aware Conflict Reasoning)", fs=9.8, c=CICR, bold=True)
box(ax, 58.2, 38.2, 18.6, 4.8, "#FFF8E1", CICR, lw=1.4, r=0.12, z=5)
text(ax, 67.5, 40.6, "d = |s_m - s_ref,m|, top-k -> con/conf masks", fs=9.0, c="#C2410C", bold=True)
box(ax, 58.2, 31.8, 18.6, 4.8, "#FFF8E1", CICR, lw=1.4, r=0.12, z=5)
text(ax, 67.5, 34.2, "C = JS(P_con || P_conf)", fs=8.2, c="#C2410C", bold=True)
text(ax, 67.5, 29.2, "Outputs: C, con_masks, conf_masks", fs=9.0, c="#C2410C", italic=True)

# flow arrows between modules
# Token-level Sentiment Projection -> IEC (on vision tokens)
arrow(ax, 40.5, yv, 57.0, yv, c=CV, lw=2.2)
text(ax, 48.8, yv + 4.0, "H_V  (for IEC)", fs=10.5, c=CV, bold=True, italic=True)

# A-Module outputs (senti_ref) -> ICR only
arrow(ax, 54.5, 36.0, 57.0, 36.0, c=CP, lw=2.0)
text(ax, 56.4, 39.3, "senti_ref_{T,V,A} -> ICR", fs=11.0, c="#9D174D", bold=True, italic=True)

# IEC -> ICR: ICR operates on compressed tokens
arrow(ax, 67.5, 58.0, 67.5, 49.5, c=CIEC, lw=2.0)
text(ax, 71.0, 53.4, "on pruned H_V^IEC / H_A", fs=9.5, c=CIEC, italic=True, ha="left")

# fusion
box(ax, 80.8, 24.0, 10.8, 56.0, CFUSL, CFUS, lw=2.6, r=0.3, z=2)
text(ax, 86.2, 75.0, "C-Driven Dynamic Fusion", fs=12.8, c=CFUS, bold=True)
text(ax, 86.2, 70.8, "(DyRoutTrans)", fs=10.5, c=CFUS, bold=True)
text(ax, 86.2, 66.8, "Inputs: H_T, H_A, H_V^IEC", fs=9.8, c="#4C1D95", italic=True)
text(ax, 86.2, 64.0, "Gate/Router: C + con/conf masks", fs=9.8, c="#4C1D95", italic=True)
box(ax, 81.8, 54.0, 8.8, 7.6, "#D1C4E9", CFUS, lw=1.4, r=0.14, z=4)
text(ax, 86.2, 57.8, "Conflict Branch", fs=8.5, c="#4C1D95", bold=True)
box(ax, 81.8, 44.2, 8.8, 7.6, "#D1C4E9", CFUS, lw=1.4, r=0.14, z=4)
text(ax, 86.2, 48.0, "Complement Branch", fs=8.5, c="#4C1D95", bold=True)
text(ax, 86.2, 35.2, "route by C / masks", fs=9.0, c="#4C1D95", italic=True)

# branch -> fusion arrows
arrow(ax, 77.9, 67.0, 80.7, 72.8, c=CV, lw=2.3, rad=0.0)         # IEC -> Fusion
text(ax, 78.8, 76.2, "H_V^IEC", fs=12.0, c=CV, bold=True, italic=True, ha="left")
arrow(ax, 77.9, 36.0, 84.2, 36.0, c=CICR, lw=2.2, rad=0.0)       # C/masks
text(ax, 79.2, 39.6, "C + con/conf masks", fs=11.4, c=CICR, bold=True, italic=True, ha="left")
arrow(ax, 78.4, 55.0, 80.7, 55.0, c=CT, lw=1.8, rad=0.0)         # local feature input
arrow(ax, 78.4, 49.0, 80.7, 49.0, c=CA, lw=1.8, rad=0.0)         # local feature input
text(ax, 78.6, 51.6, "H_T, H_A", fs=11.0, c="#334155", bold=True, italic=True, ha="left")

# output
box(ax, 92.5, 40.0, 5.8, 24.0, COUTL, COUT, lw=2.4, r=0.25, z=3)
text(ax, 95.4, 58.0, "SentiCLS", fs=11, c=COUT, bold=True)
text(ax, 95.4, 53.8, "(MLP)", fs=9.0, c=COUT, italic=True)
box(ax, 92.8, 43.0, 5.2, 6.0, "#DBEAFE", COUT, lw=1.4, r=0.14, z=4)
text(ax, 95.4, 45.9, "y_hat", fs=12, c=COUT, bold=True)
text(ax, 95.4, 41.3, "Sentiment Score", fs=8.8, c=COUT)
arrow(ax, 91.7, 52.0, 92.4, 52.0, c=CFUS, lw=2.2)


out_path = "/home/yechenlu/KuDA/results/kuda_conflict_framework_final.png"
plt.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")

