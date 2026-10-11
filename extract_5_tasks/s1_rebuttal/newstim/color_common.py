# color_common.py
# S1/S3 신규 자극 생성에 공용으로 쓰는 색 변환/렌더링 유틸.
#
# S1 원본 렌더링 코드는 저장소 전체(build/, vlm_figure/, 홈 디렉토리, VSCode 로컬 히스토리)를
# 뒤졌지만 존재하지 않았다 — 일회성 스크립트가 소실된 것으로 보임(inf_curriculum_*.csv 생성
# 스크립트가 없는 것과 같은 패턴). 대신 기존 78쌍 PNG를 픽셀 단위로 역분석해 렌더링 방식을
# 검증했다:
#   - 224x224, 배경/여백 없음, 좌우 정확히 절반씩 hex1(왼쪽)/hex2(오른쪽) 단색
#   - 경계(x=110~114)에 약 3~4px 폭의 안티에일리어싱 블렌드
#   - "4배(896x896) 캔버스에 좌우 하드 스플릿 → LANCZOS로 224x224 축소"로 재현했더니
#     원본과 max diff=1/255, mean diff=0.006 (2800ff_0028ff.png 기준) — 사실상 동일
# 이 render_pair()가 그 검증된 재현 렌더러다.

import numpy as np
from PIL import Image
from skimage import color as skcolor


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "{:02x}{:02x}{:02x}".format(*[int(round(c)) for c in rgb])


def clamp255(v):
    return max(0, min(255, int(round(v))))


def render_pair(hex1, hex2, size=224, scale=4):
    """S1 좌우 반반 컬러 페어 이미지. 원본 렌더링 픽셀 검증 완료 (본 파일 docstring 참고)."""
    big = size * scale
    arr = np.zeros((big, big, 3), dtype=np.uint8)
    arr[:, :big // 2] = hex_to_rgb(hex1)
    arr[:, big // 2:] = hex_to_rgb(hex2)
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)


def render_chip(hexcode, size=336):
    """S3 대상 칩 이미지 — 원본 확인 결과 단색 사각형 1장뿐(참조 15색은 프롬프트 텍스트로만 제공,
    이미지에는 없음). 그대로 단색 렌더링."""
    rgb = hex_to_rgb(hexcode)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return Image.fromarray(arr)


def srgb_to_lab(hexcode):
    rgb = np.array(hex_to_rgb(hexcode), dtype=np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb.reshape(1, 1, 3))
    return lab.reshape(3)


def ciede2000(hex1, hex2):
    lab1 = srgb_to_lab(hex1).reshape(1, 1, 3)
    lab2 = srgb_to_lab(hex2).reshape(1, 1, 3)
    return float(skcolor.deltaE_ciede2000(lab1, lab2)[0, 0])


# sRGB 감마트 내 이론적 최대 CIEDE2000 거리 = 순흑 vs 순백 (자연스러운 정규화 기준점,
# 특정 쌍 집합에 의존하지 않아 원본 78쌍/신규 쌍 모두에 동일하게 적용 가능)
D_REF = ciede2000("000000", "ffffff")


def ciede2000_sim(hex1, hex2):
    """[0,1] 정규화된 유사도 proxy = 1 - min(deltaE00/D_REF, 1). D_REF=흑백 CIEDE2000."""
    d = ciede2000(hex1, hex2)
    return 1.0 - min(d / D_REF, 1.0)


if __name__ == "__main__":
    print("D_REF (black-white CIEDE2000) =", D_REF)
