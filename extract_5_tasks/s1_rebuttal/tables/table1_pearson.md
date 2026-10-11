### 표 1. S1 예측–정답 Pearson r (fwd / rev), S1 단계 종료 체크포인트

"이미지만" = 원본 이미지 + hex 제거 프롬프트 (c_nohex 조건)
"hex만" = 회색 이미지 + hex 포함 프롬프트 (a_text_only 조건)

| model | 이미지만 (fwd) | 이미지만 (rev) | hex만 (fwd) | hex만 (rev) |
|---|---|---|---|---|
| LLaVA   | 0.451 | 0.601 | 0.527 | 0.420 |
| VILA-3B | 0.629 | 0.840 | 0.937 | 0.775 |

출처: extract_5_tasks/result_s1_rebuttal/{llava,vila_3b}/_table_v2.md
(c_nohex / a_text_only 조건, S1_fwd / S1_rev 체크포인트, score_v2.py 산출)
