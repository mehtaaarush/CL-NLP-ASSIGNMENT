"""Builds LSTM_Practice_Solutions.pdf. Run with: python build_pdf.py"""
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONTS = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Body", FONTS + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", FONTS + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", FONTS + "ariali.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-Bold")

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
RULE = colors.HexColor("#c5ced8")
TINT = colors.HexColor("#eef3f8")

BODY = ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=15, textColor=INK, spaceAfter=5)
EQ = ParagraphStyle("eq", parent=BODY, leftIndent=16, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Body-Bold", fontSize=19, leading=24,
                       textColor=ACCENT, spaceAfter=2)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Body-Bold", fontSize=13, leading=17,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Body-Bold", spaceBefore=6, spaceAfter=3,
                    keepWithNext=1)
CELL = ParagraphStyle("cell", parent=BODY, spaceAfter=0, alignment=TA_CENTER)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Body-Bold")


def fmt(text):
    """Turns x_t / x_(t-1) and e^z / e^(-z) notation into sub/superscripts."""
    text = re.sub(r"_\(([^)]*)\)", r"<sub>\1</sub>", text)
    text = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", text)
    text = re.sub(r"\^\(([^)]*)\)", r"<super>\1</super>", text)
    return re.sub(r"\^([A-Za-z0-9.]+)", r"<super>\1</super>", text)


def p(text):
    return Paragraph(fmt(text), BODY)


def eq(text):
    return Paragraph(fmt(text), EQ)


def h1(text):
    return Paragraph(text, H1)


def h2(text):
    return Paragraph(fmt(text), H2)


def table(rows, widths):
    data = [[Paragraph(fmt(str(c)), HEAD if r == 0 else CELL) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), TINT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


story = [
    Paragraph("LSTM Practice Questions: Solutions", TITLE),
    Spacer(1, 6),
    p("I used the sigmoid σ(z) = 1 / (1 + e^(−z)), the tanh formula given in the instructions, and "
      "rounded intermediate values to 4 decimal places as asked."),

    h1("Question 1: Single-cell forward pass"),
    p("The plan is to find each gate's input z first, push it through the right activation, and "
      "then update the cell state before the hidden state, because h_t depends on C_t."),
    p("Given: x_t = 0.5, h_(t−1) = 0.2, C_(t−1) = 0.4."),
    h2("Forget gate"),
    eq("z = W_f·x_t + U_f·h_(t−1) + b_f = 0.7(0.5) + 0.3(0.2) + 0.1 = 0.35 + 0.06 + 0.1 = 0.51"),
    eq("f_t = σ(0.51) = 1 / (1 + e^(−0.51)) = 1 / (1 + 0.6005) = 1 / 1.6005 = <b>0.6248</b>"),
    h2("Input gate"),
    eq("z = 0.6(0.5) + 0.4(0.2) + 0.0 = 0.30 + 0.08 = 0.38"),
    eq("i_t = σ(0.38) = 1 / (1 + e^(−0.38)) = 1 / (1 + 0.6839) = 1 / 1.6839 = <b>0.5939</b>"),
    h2("Candidate cell state"),
    p("The candidate uses tanh rather than sigmoid, so it can be negative as well as positive."),
    eq("z = 0.9(0.5) + 0.2(0.2) + 0.0 = 0.45 + 0.04 = 0.49"),
    eq("C̃_t = tanh(0.49) = (e^0.49 − e^(−0.49)) / (e^0.49 + e^(−0.49))"
       " = (1.6323 − 0.6126) / (1.6323 + 0.6126) = 1.0197 / 2.2449 = <b>0.4542</b>"),
    h2("Cell state update"),
    p("The forget gate decides how much of the old state stays, and the input gate decides how "
      "much of the candidate is added:"),
    eq("C_t = f_t·C_(t−1) + i_t·C̃_t = 0.6248(0.4) + 0.5939(0.4542) = 0.2499 + 0.2697 = 0.51967"
       " ≈ <b>0.5197</b>"),
    p("(I kept the extra digits of each product here, 0.24992 and 0.26975, and rounded only the "
      "sum.)"),
    h2("Output gate"),
    eq("z = 0.5(0.5) + 0.5(0.2) + 0.1 = 0.25 + 0.10 + 0.10 = 0.45"),
    eq("o_t = σ(0.45) = 1 / (1 + e^(−0.45)) = 1 / (1 + 0.6376) = 1 / 1.6376 = <b>0.6106</b>"),
    h2("Hidden state"),
    eq("tanh(C_t) = tanh(0.5197) = (1.6815 − 0.5947) / (1.6815 + 0.5947) = 1.0868 / 2.2762 = 0.4775"),
    eq("h_t = o_t · tanh(C_t) = 0.6106 × 0.4775 = <b>0.2916</b>"),
    p("Without rounding along the way this comes out as 0.2915, so the small difference is only "
      "from the 4 d.p. rounding."),
    KeepTogether([
        h2("Summary"),
        table([["f_t", "i_t", "C̃_t", "C_t", "o_t", "h_t"],
               ["0.6248", "0.5939", "0.4542", "0.5197", "0.6106", "0.2916"]], [2.6 * cm] * 6),
    ]),

    h1("Question 2: Parameter count"),
    p("Let m = 10 be the input size and n = 32 the hidden size. Each of the four gates (forget, "
      "input, candidate, output) has its own weights:"),
    eq("W (input to hidden): n × m = 32 × 10 = 320"),
    eq("U (hidden to hidden): n × n = 32 × 32 = 1024"),
    eq("b (one bias vector): n = 32"),
    h2("(a) One bias vector per gate"),
    eq("Per gate: 320 + 1024 + 32 = 1376"),
    eq("Total: 4 × 1376 = <b>5504 parameters</b>"),
    h2("(b) Two bias vectors per gate (PyTorch)"),
    p("The only change is that each gate has 32 more bias values."),
    eq("Per gate: 320 + 1024 + 32 + 32 = 1408"),
    eq("Total: 4 × 1408 = <b>5632 parameters</b>"),
    p("As a check, the difference is 4 × 32 = 128, and 5504 + 128 = 5632. I also confirmed this by "
      "counting the parameters of nn.LSTM(10, 32) in PyTorch."),

    h1("Question 3: Vanishing gradient through the cell state"),
    h2("(a) f = 0.9"),
    eq("k = 10: 0.9^10 = <b>0.3487</b>"),
    eq("k = 50: 0.9^50 = <b>0.0052</b> (more precisely 0.00515)"),
    h2("(b) k = 50"),
    eq("f = 0.5: 0.5^50 = <b>8.88 × 10^(−16)</b>, which is effectively zero"),
    eq("f = 0.99: 0.99^50 = <b>0.6050</b>"),
    h2("(c) What this means"),
    p("A dependency 50 steps back can only be learned if enough gradient survives the trip, so with "
      "f = 0.5 the signal is gone, with f = 0.9 only about 0.5% is left, and with f = 0.99 about 60% "
      "is still there. In a vanilla RNN the gradient is repeatedly multiplied by weights and "
      "activation derivatives that it has no control over, so it usually shrinks quickly. In an "
      "LSTM the cell state is updated additively and the forget gate can be learned to stay close "
      "to 1, which keeps a much cleaner path for the gradient across long gaps."),

    h1("Question 4: Electricity load forecasting"),
    h2("(a) Hourly records"),
    eq("3 years × 365 days × 24 hours = <b>26,280 records</b>"),
    h2("(b) Windows with stride 1"),
    p("Each window needs 168 input hours plus 24 output hours, so 192 consecutive hours. A window "
      "can start anywhere as long as all 192 hours fit, which gives:"),
    eq("26,280 − 192 + 1 = <b>26,089 windows</b>"),
    p("The +1 is there because a dataset of exactly 192 records would still give one window."),
    h2("(c) Tensor shapes"),
    p("Using (batch, time steps, features):"),
    eq("Input batch: <b>(64, 168, 3)</b>"),
    eq("Output batch: <b>(64, 24)</b>, since only load is predicted, for 24 hours. This is the same "
       "as (64, 24, 1) if the target keeps a feature dimension."),
    h2("(d) Training windows and mini-batches"),
    eq("Training records: 0.70 × 26,280 = 18,396"),
    p("The split is on records, so a training window must fit completely inside those records. "
      "Otherwise future test data would leak into training."),
    eq("Training windows: 18,396 − 192 + 1 = <b>18,205</b>"),
    eq("Batches: 18,205 / 64 = 284.45, so 284 full batches and 1 partial batch of 29 windows"),
    eq("Mini-batches per epoch = <b>285</b>"),
    h2("(e) LSTM parameters"),
    p("Same formula as Question 2, with m = 3, n = 64 and one bias vector per gate:"),
    eq("4 × (64 × 3 + 64 × 64 + 64) = 4 × (192 + 4096 + 64) = 4 × 4352 = <b>17,408 parameters</b>"),

    h1("Question 5: Predictive maintenance"),
    h2("(a) Min-max normalisation"),
    eq("x_norm = (x − x_min) / (x_max − x_min) = (95 − 20) / (120 − 20) = 75 / 100 = <b>0.75</b>"),
    h2("(b) Prediction"),
    p("0.83 is greater than the 0.5 threshold, so the model predicts <b>failure</b> within the next "
      "cycle."),
    h2("(c) Confusion matrix and metrics"),
    p("Starting from the 30 correct flags, the other counts follow from the 45 flags, the 40 actual "
      "failures and the 1,000 cycles:"),
    eq("TP = 30 (correct flags)"),
    eq("FP = 45 − 30 = 15 (flagged, but no failure)"),
    eq("FN = 40 − 30 = 10 (real failures that were missed)"),
    eq("TN = 1000 − 30 − 15 − 10 = 945"),
    Spacer(1, 3),
    table([["", "Predicted failure", "Predicted OK", "Total"],
           ["Actual failure", "TP = 30", "FN = 10", "40"],
           ["Actual OK", "FP = 15", "TN = 945", "960"],
           ["Total", "45", "955", "1000"]], [3.6 * cm, 3.6 * cm, 3.6 * cm, 2.4 * cm]),
    Spacer(1, 6),
    eq("Precision = TP / (TP + FP) = 30 / 45 = <b>0.6667</b>"),
    eq("Recall = TP / (TP + FN) = 30 / 40 = <b>0.75</b>"),
    eq("F1 = 2 × P × R / (P + R) = 2(0.6667)(0.75) / (0.6667 + 0.75) = 1.0000 / 1.4167 = <b>0.7059</b>"),
    eq("Accuracy = (TP + TN) / total = (30 + 945) / 1000 = <b>0.975</b>"),
    h2("(d) Which metric matters"),
    p("I think the team should prioritise <b>recall</b>. A missed failure (FN) can mean an unplanned "
      "breakdown and lost production, which usually costs far more than an unnecessary inspection "
      "(FP). The current recall of 0.75 means 1 in 4 failures is still missed. Precision should "
      "still be watched so that there are not so many false alarms that people stop trusting the "
      "model, and F1 is a useful way to track both."),
    p("Accuracy is misleading because the classes are very imbalanced, with only 4% of cycles being "
      "failures. A useless model that always says \"no failure\" would still get 960 / 1000 = 96% "
      "accuracy while catching none of the failures, so the 97.5% here looks better than it really "
      "is."),
]

SimpleDocTemplate("LSTM_Practice_Solutions.pdf", pagesize=A4, leftMargin=2.2 * cm,
                  rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="LSTM Practice Questions: Solutions").build(story)
