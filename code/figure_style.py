"""Common figure style for the P4 series: TeX Gyre Pagella (Palatino clone),
8 pt base size, U+2212 minus sign, TrueType embedding."""
from pathlib import Path
import matplotlib
from matplotlib import font_manager

CACHE = Path(__file__).resolve().parent / ".font_cache"
OTF_DIRS = [Path("/usr/share/texmf/fonts/opentype/public/tex-gyre"),
            Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre")]
FAMILY = "TeX Gyre Pagella"

PALETTE = ["#1f4e79", "#c55a11", "#2e7d32", "#7b1fa2", "#00838f", "#9e9d24", "#6d6d6d"]


def _otf_to_ttf(src, dst):
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    font = TTFont(src)
    gs = font.getGlyphSet()
    glyf, order = {}, font.getGlyphOrder()
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = g = newTable("glyf")
    g.glyphOrder, g.glyphs = order, glyf
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    font["maxp"] = m = font["maxp"]
    m.tableVersion = 0x00010000
    for a in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs",
              "maxInstructionDefs", "maxStackElements", "maxSizeOfInstructions",
              "maxComponentElements", "maxPoints", "maxContours",
              "maxCompositePoints", "maxCompositeContours", "maxComponentDepth"):
        setattr(m, a, 0)
    m.maxZones = 1
    font["head"].indexToLocFormat = 0
    font["head"].glyphDataFormat = 0
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = None
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)


def _ensure_fonts():
    CACHE.mkdir(exist_ok=True)
    fm = font_manager.fontManager
    fm.ttflist = [f for f in fm.ttflist
                  if not (f.fname.lower().endswith(".otf") and "pagella" in f.fname.lower())]
    for style in ("regular", "bold", "italic", "bolditalic"):
        dst = CACHE / f"texgyrepagella-{style}.ttf"
        if not dst.exists():
            src = next((d / f"texgyrepagella-{style}.otf" for d in OTF_DIRS
                        if (d / f"texgyrepagella-{style}.otf").exists()), None)
            if src is None:
                raise FileNotFoundError("TeX Gyre Pagella OTF not found")
            _otf_to_ttf(str(src), str(dst))
        font_manager.fontManager.addfont(str(dst))


def aplicar():
    _ensure_fonts()
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": [FAMILY], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
        "mathtext.fontset": "custom", "mathtext.rm": FAMILY,
        "mathtext.it": FAMILY + ":italic", "mathtext.bf": FAMILY + ":bold",
        "mathtext.sf": FAMILY, "mathtext.cal": FAMILY,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.4,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
