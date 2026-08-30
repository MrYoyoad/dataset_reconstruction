"""Paths, palette, layout constants and the figure dictionary for the 2026-08-31 deck.

Theme: white / Cambria headings / Calibri body — deliberately the SAME look as the
2026-05-14 deck (v18) so the two decks read as one series (deviation from the dark
style_guide/pptx.md template is logged in docs/presentation-remarks-log.md).
"""
import os
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIG_DIR = os.path.join(ROOT, "figures", "deck_2026_08_31")      # clean deck figures
EQ_DIR = os.path.join(FIG_DIR, "eq")                             # rendered equations
OUT_PPTX = os.path.join(ROOT, "notes", "supervisor_meeting_2026_08_31.pptx")
OUT_COPY = os.path.join(ROOT, "figures", "supervisor_meeting_2026_08_31_v1.pptx")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

MEETING_DATE = "2026-08-31"
FOOTER = f"Yoad Oxman / Gal Vardi  -  {MEETING_DATE}"

# ----- palette (May deck) -----
BLACK = RGBColor(0x00, 0x00, 0x00)
GRAY = RGBColor(0x55, 0x55, 0x55)
LGRAY = RGBColor(0x88, 0x88, 0x88)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x1F, 0x4E, 0x79)       # primary accent
RED = RGBColor(0xC0, 0x39, 0x2B)        # headline / kinked
GREEN_OK = RGBColor(0x2E, 0x7D, 0x32)   # positive / smooth
AMBER = RGBColor(0xB2, 0x6A, 0x00)      # open / caveat
LIGHTBLUE_FILL = RGBColor(0xEA, 0xF0, 0xF7)
LIGHTRED_FILL = RGBColor(0xFA, 0xEC, 0xEA)
LIGHTGREEN_FILL = RGBColor(0xEA, 0xF5, 0xEA)
LIGHTGRAY_FILL = RGBColor(0xF3, 0xF3, 0xF3)

# matplotlib hex twins (figures must match the deck)
HEX_BLUE, HEX_RED, HEX_GREEN, HEX_GRAY, HEX_AMBER = "#1F4E79", "#C0392B", "#2E7D32", "#555555", "#B26A00"
HEX_ORANGE = "#E07B39"

# ----- fonts -----
HEAD = "Cambria"
BODY = "Calibri"

# ----- geometry: 16:9 -----
SL_W = Inches(13.333)
SL_H = Inches(7.5)
MX = Inches(0.55)          # horizontal margin
MY = Inches(0.35)          # top margin
CT = Inches(1.30)          # content top (below title rule)
CW = SL_W - 2 * MX         # content width
CB = Inches(6.85)          # content bottom (footer safe zone, guardrail O4)
CH = CB - CT
GAP = Inches(0.25)         # standard gutter (S1/S2)

# ----- font sizes (guardrail T1 minima respected) -----
SZ_TITLE = 28
SZ_LEAD = 18       # the one line under the title
SZ_BODY = 16
SZ_SMALL = 12
SZ_FOOT = 9

# ----- figure dictionary: slide -> file (built by make_deck_figures.py) -----
def fig(name):
    return os.path.join(FIG_DIR, name)

def eq(name):
    return os.path.join(EQ_DIR, name)

# existing assets used as-is
ASSET = {
    "di_N4": os.path.join(FIGURES, "direct_inversion", "di_grid_N4_r8_gelu_T10.png"),
    "di_N10": os.path.join(FIGURES, "direct_inversion", "di_grid_N10_r8_gelu_T10.png"),
    "faces": os.path.join(FIGURES, "phase0", "n3_three_faces.png"),
    "vit_face": os.path.join(FIGURES, "phase0", "phase0_full_r8_n1.png"),
}
