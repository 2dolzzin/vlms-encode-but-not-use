## 회색(gray) 조건 raw 출력 분포 — 표 A 각주용

### S3 (색 이름 응답, parse_color 적용)

| model | ckpt | n | top-3 (비율) | gray 비율 |
|---|---|---|---|---|
| llava | Base | 330 | gray(1.00) | 1.000 |
| llava | Final_fwd | 330 | gray(1.00) | 1.000 |
| llava | Final_rev | 330 | gray(1.00) | 1.000 |
| mipha | Base | 330 | gray(1.00) | 1.000 |
| mipha | Final_fwd | 330 | gray(1.00) | 1.000 |
| mipha | Final_rev | 330 | gray(1.00) | 1.000 |
| vila_3b | Base | 330 | beige(1.00) | 0.000 |
| vila_3b | Final_fwd | 330 | gray(1.00) | 1.000 |
| vila_3b | Final_rev | 330 | gray(1.00) | 1.000 |
| vila_8b | Base | 330 | gray(1.00) | 1.000 |
| vila_8b | Final_fwd | 330 | gray(1.00) | 1.000 |
| vila_8b | Final_rev | 330 | gray(1.00) | 1.000 |
| openflamingo | Base | 330 | unknown(1.00) | 0.000 |
| openflamingo | Final_fwd | 330 | gray(1.00) | 1.000 |
| openflamingo | Final_rev | 330 | gray(1.00) | 1.000 |

### S2 (숫자 응답, parse_digit 적용)

| model | ckpt | n | top-3 (비율) | 상수? |
|---|---|---|---|---|
| llava | Base | 140 | 0(1.00) | 예 |
| llava | Final_fwd | 140 | 9(1.00) | 예 |
| llava | Final_rev | 140 | 9(1.00) | 예 |
| mipha | Base | 140 | 1(1.00) | 예 |
| mipha | Final_fwd | 140 | 3(1.00) | 예 |
| mipha | Final_rev | 140 | 0(1.00) | 예 |
| vila_3b | Base | 140 | 6(1.00) | 예 |
| vila_3b | Final_fwd | 140 | 0(1.00) | 예 |
| vila_3b | Final_rev | 140 | 0(1.00) | 예 |
| vila_8b | Base | 140 | 0(1.00) | 예 |
| vila_8b | Final_fwd | 140 | 0(1.00) | 예 |
| vila_8b | Final_rev | 140 | 0(1.00) | 예 |
| openflamingo | Base | 140 | -1(1.00) | 예 |
| openflamingo | Final_fwd | 140 | 1(1.00) | 예 |
| openflamingo | Final_rev | 140 | 0(1.00) | 예 |

### S0 (원시 출력 문자열 기준 상수 여부)

| model | ckpt | n | 고유 출력 수 | 상수? | 최빈 출력(비율) |
|---|---|---|---|---|---|
| llava | Base | 1800 | 3 | 아니오 | 'B'(0.67) |
| llava | Final_fwd | 1800 | 2 | 아니오 | 'Answer: GRAV'(0.83) |
| llava | Final_rev | 1800 | 3 | 아니오 | 'Answer: GRAV'(0.67) |
| mipha | Base | 1800 | 4 | 아니오 | 'E I O P'(0.50) |
| mipha | Final_fwd | 1800 | 4 | 아니오 | 'Output: ZWQVNC'(0.50) |
| mipha | Final_rev | 1800 | 6 | 아니오 | 'Output: QXBJYZ'(0.17) |
| vila_3b | Base | 1800 | 2 | 아니오 | 'a, b, c, d, f, h'(0.83) |
| vila_3b | Final_fwd | 1800 | 6 | 아니오 | 'Output: QQZIVP'(0.17) |
| vila_3b | Final_rev | 1800 | 5 | 아니오 | 'Answer: QQZIV'(0.33) |
| vila_8b | Base | 1800 | 5 | 아니오 | 'C, D, E, F, G, H'(0.33) |
| vila_8b | Final_fwd | 1800 | 6 | 아니오 | 'Output: TIJBGL'(0.17) |
| vila_8b | Final_rev | 1800 | 6 | 아니오 | 'Answer: YKOHLN'(0.17) |
| openflamingo | Base | 1800 | 4 | 아니오 | 'You are given an image of a six-row letter chart, each row has exactly six uppercase letters. Read only the first row. Answer with exactly six letters.'(0.50) |
| openflamingo | Final_fwd | 1800 | 5 | 아니오 | 'ADJ'(0.33) |
| openflamingo | Final_rev | 1800 | 6 | 아니오 | 'HHSVZ'(0.17) |
