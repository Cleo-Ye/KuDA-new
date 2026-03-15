"""
KuDA-Conflict Framework Diagram
仿照原 KuDA 论文横版框架图风格
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── figure canvas ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 11))
ax  = fig.add_axes([0.01, 0.06, 0.98, 0.86])
ax.set_xlim(0, 26)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── palette ────────────────────────────────────────────────────────────────────
CV   = '#43A047'; CVl  = '#E8F5E9'  # vision   green
CT   = '#8E24AA'; CTl  = '#F3E5F5'  # text     purple
CA   = '#F57C00'; CAl  = '#FFF3E0'  # audio    orange
CSPE = '#F9A825'; CSPl = '#FFFDE7'  # senti-proj  yellow
CAAR = '#E91E63'; CAARl= '#FCE4EC'  # align-ref   pink
CIEC = '#388E3C'; CIECl= '#E8F5E9'  # IEC         dark-green
CICR = '#EF6C00'; CICRl= '#FFF3E0'  # ICR/ConflJS orange-dark
CFUS = '#512DA8'; CFUSl= '#EDE7F6'  # fusion      purple-dark
COUT = '#1565C0'; COUTl= '#E8EAF6'  # output      blue-dark
CLOSS= '#C62828'; CLOSSl='#FFEBEE'  # loss        red

# ── row centres ────────────────────────────────────────────────────────────────
YV, YT, YA = 8.2, 5.3, 2.4

# ── helpers ────────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=1.8, r=0.18, z=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.04,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=7.5, col='black', bold=False, italic=False,
        ha='center', va='center', z=5):
    ax.text(x, y, s, fontsize=fs, color=col,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            ha=ha, va=va, zorder=z)

def harrow(ax, x1, x2, y, col='#455A64', lw=1.8, label=None, lfs=6.2, z=6):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw), zorder=z)
    if label:
        ax.text((x1+x2)/2, y+0.22, label, ha='center', va='bottom',
                fontsize=lfs, color=col, style='italic', zorder=z+1)

def varrow(ax, x, y1, y2, col='#455A64', lw=1.8, label=None, lfs=6.2, z=6):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw), zorder=z)
    if label:
        ax.text(x+0.15, (y1+y2)/2, label, ha='left', va='center',
                fontsize=lfs, color=col, style='italic', zorder=z+1)

def curved_arrow(ax, x1, y1, x2, y2, rad=0.35, col='#455A64', lw=1.5, z=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->',
                                connectionstyle=f'arc3,rad={rad}',
                                color=col, lw=lw), zorder=z)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION DIVIDER LINES + HEADER LABELS
# ══════════════════════════════════════════════════════════════════════════════
div_xs = [1.55, 4.85, 6.45, 8.55, 10.55, 13.45, 16.45]
for dx in div_xs:
    ax.axvline(dx, ymin=0.01, ymax=0.94, color='#CFD8DC', lw=1.2,
               linestyle='--', zorder=1)

sec_headers = [
    (0.78,  'Input'),
    (3.2,   'Encoding'),
    (5.65,  'Sentiment\nProjection'),
    (7.5,   'A Module\nAlign. Ref.'),
    (9.55,  'Branch 1\nIEC (Vision)'),
    (11.95, 'Branch 2\nICR (All mod.)'),
    (14.95, 'C-Driven Dynamic Fusion'),
    (18.0,  'Output'),
]
for hx, hl in sec_headers:
    ax.text(hx, 10.35, hl, ha='center', va='top', fontsize=7.5,
            fontweight='bold', color='#263238',
            bbox=dict(boxstyle='round,pad=0.28', fc='#ECEFF1',
                      ec='#90A4AE', lw=1.1, alpha=0.95), zorder=8)

# ══════════════════════════════════════════════════════════════════════════════
# 0. INPUT COLUMN  (x: 0.1 – 1.55)
# ══════════════════════════════════════════════════════════════════════════════
input_data = [
    (YV, CV, CVl, 'Vision', 'video frames'),
    (YT, CT, CTl, 'Text',   'sentence tokens'),
    (YA, CA, CAl, 'Audio',  'audio frames'),
]
for yc, ec, fc, name, sub in input_data:
    rbox(ax, 0.1, yc-0.6, 1.3, 1.2, fc, ec, lw=2.0, r=0.25)
    txt(ax, 0.75, yc+0.2,  name, fs=9.5, col=ec, bold=True)
    txt(ax, 0.75, yc-0.25, sub,  fs=6.2, col='#546E7A', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. ENCODING  (x: 1.55 – 4.85)
#    Unimodal Encoder per modality → hidden token sequences
# ══════════════════════════════════════════════════════════════════════════════
for yc, mc, mlt, encname, sublabel in [
    (YV, CV, CVl, 'Feat.Ext.\n+ Transformer', 'H_V ∈ ℝᴸᵛˣᴰ'),
    (YT, CT, CTl, 'BERT\nEncoder',            'H_T ∈ ℝᴸᵗˣᴰ'),
    (YA, CA, CAl, 'Feat.Ext.\n+ Transformer', 'H_A ∈ ℝᴸᵃˣᴰ'),
]:
    harrow(ax, 1.4, 1.65, yc, col=mc)
    # Wider encoder box fills the section
    rbox(ax, 1.65, yc-0.52, 2.7, 1.04, mlt, mc, lw=2.0)
    txt(ax, 3.0, yc+0.18, encname,  fs=8,   col=mc, bold=True)
    txt(ax, 3.0, yc-0.26, sublabel, fs=6.5, col=mc, italic=True)
    # forward arrow to Sentiment Projection
    harrow(ax, 4.35, 4.6, yc, col=mc, lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT PROJECTION  (x: 4.85 – 6.45)
# ══════════════════════════════════════════════════════════════════════════════
for yc, mc in [(YV, CV), (YT, CT), (YA, CA)]:
    rbox(ax, 4.6, yc-0.48, 1.65, 0.96, CSPl, CSPE, lw=1.6)
    txt(ax, 5.425, yc+0.15, 'Sentiment\nProjector', fs=7, col='#F57F17', bold=True)
    txt(ax, 5.425, yc-0.27, 'pₘ∈ℝᴸˣᶜ, sₘ∈ℝᴸ', fs=6.2, col='#E65100', italic=True)
    harrow(ax, 6.25, 6.5, yc, col=mc, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# 3. A MODULE – AlignmentAwareReference  (x: 6.45 – 8.55, span all rows)
# ══════════════════════════════════════════════════════════════════════════════
AX, AY, AW = 6.45, YA-1.05, 2.05
AH = YV - YA + 2.1
rbox(ax, AX, AY, AW, AH, CAARl, CAAR, lw=2.8, r=0.35, z=2)

txt(ax, AX+AW/2, YV+0.65,  'A Module',              fs=9.5, col=CAAR, bold=True)
txt(ax, AX+AW/2, YV+0.28,  'AlignmentAwareRef.',    fs=7.5, col=CAAR, bold=True)
txt(ax, AX+AW/2, YT+0.65,  'V→T, A→T Cross-Attn',  fs=6.5, col='#880E4F', italic=True)
txt(ax, AX+AW/2, YT+0.25,  'Rel_T reweighting',     fs=6.5, col='#880E4F', italic=True)
txt(ax, AX+AW/2, YT-0.18,  'ã_{ij}=a_{ij}·Rel_T(j)',fs=6.2, col='#880E4F', italic=True)
txt(ax, AX+AW/2, YA+0.50,  '→ senti_ref_{T,V,A}',  fs=7,   col='#880E4F', bold=True)
txt(ax, AX+AW/2, YA+0.12,  '→ attn_vt (shared)',    fs=6.5, col='#880E4F', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. B MODULE – IEC TextGuidedVisionPruner  (x: 8.55 – 10.55, Vision row only)
# ══════════════════════════════════════════════════════════════════════════════
BX, BW = 8.55, 1.95
rbox(ax, BX, YV-0.72, BW, 1.44, CIECl, CIEC, lw=2.6, r=0.22)
txt(ax, BX+BW/2, YV+0.38,  'B Module: IEC',              fs=9,   col=CIEC, bold=True)
txt(ax, BX+BW/2, YV+0.06,  'TextGuided VisionPruner',   fs=7,   col=CIEC, bold=True)
txt(ax, BX+BW/2, YV-0.22,  'Input: H_V + attn_vt',      fs=6,   col='#1B5E20', italic=True)
txt(ax, BX+BW/2, YV-0.48,  'score = attn_vt · |s_T|',   fs=5.8, col='#1B5E20', italic=True)
txt(ax, BX+BW/2, YV-0.70,  'Top-K Prune + Merge',        fs=5.8, col='#1B5E20', italic=True)

# Pre-declare ICR and Fusion x coords so parallel-branch arrows can reference them
IX, IY, IW = 10.55, YA-1.05, 2.85
IH = YV - YA + 2.1
FX, FY, FW = 13.45, YA-1.05, 2.95
FH = YV - YA + 2.1

# ── parallel branch lanes (dashed outline, no fill to avoid visual overlap) ─
rbox(ax, BX, IY, BW, IH, 'none', CIEC, lw=1.6, r=0.26, z=1)
rbox(ax, IX, IY, IW, IH, 'none', CICR, lw=1.6, r=0.26, z=1)
rbox(ax, 8.5, YV+0.95, 4.95, 0.55, '#F5F5F5', '#78909C', lw=1.2, r=0.2, z=1)
txt(ax, 8.5+4.95/2, YV+1.22, 'Two Parallel Branches', fs=7.5, col='#37474F', bold=True)
txt(ax, BX+BW/2, YV+0.86, 'Branch-1 Lane (IEC)', fs=6.2, col=CIEC, bold=True)
txt(ax, IX+IW/2, YV+0.86, 'Branch-2 Lane (ICR)', fs=6.2, col=CICR, bold=True)

# A → IEC: attn_vt (Vision row, straight arrow)
harrow(ax, AX+AW, BX, YV, col=CV, lw=2.0, label='attn_vt')

# A → ICR: senti_ref_T and senti_ref_A (straight arrows)
harrow(ax, AX+AW, IX, YT, col=CT, lw=1.6, label='senti_ref_T')
harrow(ax, AX+AW, IX, YA, col=CA, lw=1.6, label='senti_ref_A')

# A → ICR: senti_ref_V (below IEC core box)
ax.annotate('', xy=(IX, YV-0.95), xytext=(AX+AW, YV-0.95),
            arrowprops=dict(arrowstyle='->', color=CV, lw=1.5,
                            linestyle=(0, (4, 2))), zorder=6)
txt(ax, (AX+AW+IX)/2, YV-0.72, 'senti_ref_V', fs=6, col=CV, italic=True)

# IEC → Fusion: H_V^compressed (top arc, avoid crossing ICR internals)
ax.annotate('', xy=(FX, YV+0.95), xytext=(BX+BW, YV+0.95),
            arrowprops=dict(arrowstyle='->', color=CV, lw=2.2,
                            connectionstyle='arc3,rad=0.24'), zorder=7)
txt(ax, (BX+BW+FX)/2, YV+1.42, 'H_V^compressed', fs=6.5, col=CV, italic=True, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. ICR – ConflictJS  (x: 10.55 – 13.45, span all rows)
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, IX+IW/2, YV+0.62,  'ICR Module',              fs=9.5, col=CICR, bold=True)
txt(ax, IX+IW/2, YV+0.25,  'Conflict-JS',             fs=7.5, col=CICR, bold=True)
txt(ax, IX+IW/2, YV-0.08,  'Input: posteriors + senti_ref (T,V,A)', fs=5.8, col='#BF360C', italic=True)

# EvidenceSplitter inner box
rbox(ax, IX+0.18, YT+0.35, IW-0.36, 0.72, '#FFF8E1', CICR, lw=1.3, r=0.14)
txt(ax, IX+IW/2, YT+0.71, 'EvidenceSplitter',      fs=7,   col='#E65100', bold=True)
txt(ax, IX+IW/2, YT+0.45, 'd = |s_m − senti_ref_m|', fs=5.8, col='#BF360C', italic=True)

# EvidenceLevelJS inner box
rbox(ax, IX+0.18, YT-0.55, IW-0.36, 0.7, '#FFF8E1', CICR, lw=1.3, r=0.14)
txt(ax, IX+IW/2, YT-0.2,  'EvidenceLevelJS',       fs=7,   col='#E65100', bold=True)
txt(ax, IX+IW/2, YT-0.45, 'JS(P_conf ‖ P_con)',   fs=5.8, col='#BF360C', italic=True)

txt(ax, IX+IW/2, YA+0.55, '→ con_masks, conf_masks', fs=6.5, col='#BF360C', bold=True)
txt(ax, IX+IW/2, YA+0.18, '→ C (conflict intensity)', fs=7,   col='#BF360C', bold=True)
txt(ax, IX+IW/2, IY+0.25, 'Outputs: C, con_masks, conf_masks', fs=6, col='#BF360C', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. C-DRIVEN DYNAMIC FUSION  (x: 13.45 – 16.45, span all rows)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, FX, FY, FW, FH, CFUSl, CFUS, lw=2.8, r=0.35, z=2)

txt(ax, FX+FW/2, YV+0.65, 'C-Driven Fusion',         fs=9.5, col=CFUS, bold=True)
txt(ax, FX+FW/2, YV+0.28, 'DyRoutTrans',              fs=7.5, col=CFUS, bold=True)

# Two branches
rbox(ax, FX+0.18, YT+0.32, FW-0.36, 0.66, '#D1C4E9', CFUS, lw=1.3, r=0.14)
txt(ax, FX+FW/2, YT+0.65, 'Conflict Branch',          fs=7,   col='#4527A0', bold=True)
txt(ax, FX+FW/2, YT+0.42, '(conf_mask tokens)',       fs=6,   col='#4527A0', italic=True)

rbox(ax, FX+0.18, YT-0.62, FW-0.36, 0.66, '#D1C4E9', CFUS, lw=1.3, r=0.14)
txt(ax, FX+FW/2, YT-0.29, 'Complement Branch',        fs=7,   col='#4527A0', bold=True)
txt(ax, FX+FW/2, YT-0.52, '(con_mask tokens)',        fs=6,   col='#4527A0', italic=True)

txt(ax, FX+FW/2, YA+0.55, 'α = σ(k·(C − τ))',        fs=7,   col='#311B92', italic=True)
txt(ax, FX+FW/2, YA+0.18, 'h = α·h_conf+(1−α)·h_com',fs=6.5, col='#311B92', italic=True)
txt(ax, FX+FW/2, FY+0.25, 'Route by C / masks',       fs=6.2, col='#4527A0', italic=True)

# arrows from ICR → Fusion: only T and A token sequences
# (Vision tokens go IEC→Fusion directly via the arc above)
for yc, mc in [(YT, CT), (YA, CA)]:
    harrow(ax, IX+IW, FX, yc, col=mc, lw=1.8)

# C signal curved arrow ICR bottom → Fusion bottom
ax.annotate('', xy=(FX+FW/2, FY+0.65), xytext=(IX+IW/2, IY+0.65),
            arrowprops=dict(arrowstyle='->', color=CICR, lw=2.2,
                            connectionstyle='arc3,rad=-0.25'), zorder=7)
txt(ax, 14.0, IY+0.05, 'C, con/conf masks', fs=6.8, col=CICR, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7. OUTPUT  (x: 16.45 – 20.0)
# ══════════════════════════════════════════════════════════════════════════════
OX  = 16.45
OYC = (YV + YA) / 2

harrow(ax, FX+FW, OX+0.15, OYC, col=CFUS, lw=2.2, label='F_fused')

rbox(ax, OX+0.1, OYC-0.55, 1.55, 1.1, COUTl, COUT, lw=2.2, r=0.22)
txt(ax, OX+0.875, OYC+0.22, 'SentiCLS',   fs=8.5, col=COUT, bold=True)
txt(ax, OX+0.875, OYC-0.18, '(MLP Head)', fs=7,   col=COUT, italic=True)

harrow(ax, OX+1.65, OX+2.2, OYC, col=COUT, lw=2.2)
rbox(ax, OX+2.2, OYC-0.55, 1.3, 1.1, COUTl, COUT, lw=2.2, r=0.22)
txt(ax, OX+2.85, OYC+0.2,   'ŷ',             fs=13, col=COUT, bold=True)
txt(ax, OX+2.85, OYC-0.25,  'Sentiment\nScore', fs=7, col=COUT)


# ── Data pre-processing note ─────────────────────────────────────────────────
rbox(ax, 0.05, 0.12, 2.9, 0.9, '#E0F7FA', '#00838F', lw=1.3, r=0.2)
txt(ax, 1.5, 0.72, 'Data Pre-processing', fs=7, col='#006064', bold=True)
txt(ax, 1.5, 0.42, '(1) Audio CMVN (z-score per sample)', fs=6.2, col='#00695C', italic=True)
txt(ax, 1.5, 0.18, '(2) Truncation + Padding Mask',       fs=6.2, col='#00695C', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    'KuDA-Conflict Framework  |  Multi-Modal Sentiment Analysis with IEC (B-Module) & ICR (Conflict-JS)\n'
    'Extensions over original KuDA: Sentiment Projection → Alignment-Aware Ref. (A) → Vision IEC Pruning (B) → Conflict-JS (ICR) → C-Driven Fusion',
    fontsize=10.5, fontweight='bold', color='#1A237E', y=0.995,
    bbox=dict(boxstyle='round,pad=0.4', fc='#E8EAF6', ec='#3949AB', lw=1.5, alpha=0.9)
)

out_path = '/home/yechenlu/KuDA/results/kuda_conflict_framework.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved → {out_path}")
