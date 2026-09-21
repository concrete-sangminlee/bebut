"""GitHub 소셜 미리보기(Open Graph) 이미지 생성. 1280x640, Settings > General > Social preview에 수동 업로드.

실행: .venv/bin/python docs/make_og_image.py
"""

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG = (11, 13, 16)
BORDER = (38, 43, 51)
TEXT = (231, 233, 236)
DIM = (154, 162, 173)
ACCENT = (110, 231, 183)
ACCENT2 = (125, 211, 252)

FONT_DIR = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def font(size, index=0):
    return ImageFont.truetype(FONT_DIR, size, index=index)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# border frame
d.rectangle([0, 0, W - 1, H - 1], outline=BORDER, width=2)

pad = 80

d.text((pad, 100), "BEBUT · FROM-SCRATCH KOREAN LLM", font=font(24), fill=ACCENT)

title_font = font(64, index=7)  # Bold face index in the ttc, falls back if unavailable
try:
    d.text((pad, 150), "한국어 1.2B LLM을", font=title_font, fill=TEXT)
    d.text((pad, 230), "밑바닥부터 학습했습니다", font=title_font, fill=TEXT)
except Exception:
    d.text((pad, 150), "한국어 1.2B LLM을", font=font(64), fill=TEXT)
    d.text((pad, 230), "밑바닥부터 학습했습니다", font=font(64), fill=TEXT)

d.text(
    (pad, 330),
    "토크나이저부터 트랜스포머·사전학습·SFT까지 순수 PyTorch로 직접 구현",
    font=font(26),
    fill=DIM,
)

stats = [
    ("25.2B", "학습 토큰"),
    ("1장", "GPU (RTX 6000 Ada)"),
    ("74.2%", "SFT 대화 정답률"),
]

x = pad
y = 420
for value, label in stats:
    d.text((x, y), value, font=font(44, index=7), fill=ACCENT2)
    d.text((x, y + 62), label, font=font(20), fill=DIM)
    x += 320

d.text((pad, H - 70), "github.com/concrete-sangminlee/bebut", font=font(22), fill=DIM)

img.save("docs/og-image.png")
print("saved docs/og-image.png")
