# csv_out/experiments 색인

- 열 `baseline,t1_mid,…,t5_end` = 그 런의 **학습 순서** 기준 시점. `order` 열이 실제 스테이지 순서(예 `S1→S4→S2→S0→S3`).
- `by_stage/` 는 같은 내용을 **스테이지 이름** 열(`S0_mid…S4_end`)로 배치한 것.
- variant: forward / reverse / random_14203 / random_20431 / random_31024. 빈 칸 = 결과 없음(파싱 전멸 NaN 포함).
- S1_similarity 추론·ridge 는 mae(낮을수록 좋음), 그 외는 accuracy/acc(높을수록 좋음).
- fwd/rev 의 S1 추론은 hex 프롬프트, random/Qwen 은 nohex → S1 열 직접 비교 금지.

| 파일 | 행 수 |
|---|---|
| exp1_evalloss_S0.csv | 27 |
| exp1_evalloss_S1.csv | 27 |
| exp1_evalloss_S2.csv | 27 |
| exp1_evalloss_S3.csv | 27 |
| exp1_evalloss_S4.csv | 27 |
| exp1_inference_S0_logamr.csv | 27 |
| exp1_inference_S1_similarity_mae.csv | 27 |
| exp1_inference_S2_ishihara.csv | 27 |
| exp1_inference_S3_naming.csv | 27 |
| exp1_inference_S4_depth.csv | 27 |
| exp2_probe_mlp_S0_logamr.csv | 162 |
| exp2_probe_mlp_S2_ishihara.csv | 162 |
| exp2_probe_mlp_S3_naming.csv | 162 |
| exp2_probe_mlp_S4_depth.csv | 162 |
| exp2_probe_ridge_S1_similarity.csv | 162 |
| exp2_probe_linear_S0_logamr.csv | 162 |
| exp2_probe_linear_S2_ishihara.csv | 162 |
| exp2_probe_linear_S3_naming.csv | 162 |
| exp2_probe_linear_S4_depth.csv | 162 |
| exp2_probe_logreg_S0_logamr.csv | 10 |
| exp2_probe_logreg_S2_ishihara.csv | 10 |
| exp2_probe_logreg_S3_naming.csv | 10 |
| exp2_probe_logreg_S4_depth.csv | 10 |
| exp3_inference_circled_letter.csv | 27 |
| exp3_inference_touching_circles.csv | 27 |
| exp3_inference_nested_squares.csv | 27 |
| exp3_inference_olympic_circles.csv | 27 |
| exp3_inference_olympic_pentagons.csv | 27 |
| exp3_inference_subway_connections.csv | 27 |
| exp3_inference_counting_grid_blank.csv | 27 |
| exp3_inference_counting_grid_word.csv | 27 |
| exp3_inference_line_plot.csv | 27 |
| exp3_inference_mmvp.csv | 27 |
| exp3_inference_mmvp_pair_accuracy.csv | 27 |
| exp4_probe_mlp_circled_letter.csv | 162 |
| exp4_probe_mlp_counting_grid_blank.csv | 162 |
| exp4_probe_mlp_counting_grid_word.csv | 162 |
| exp4_probe_mlp_nested_squares.csv | 162 |
| exp4_probe_mlp_olympic_circles.csv | 162 |
| exp4_probe_mlp_olympic_pentagons.csv | 162 |
| exp4_probe_mlp_subway_connections.csv | 162 |
| exp4_probe_mlp_touching_circles.csv | 162 |
| exp4_probe_mlp_line_plot.csv | 162 |
| exp4_probe_mlp_mmvp.csv | 162 |
| exp4_probe_linear_circled_letter.csv | 162 |
| exp4_probe_linear_counting_grid_blank.csv | 162 |
| exp4_probe_linear_counting_grid_word.csv | 162 |
| exp4_probe_linear_nested_squares.csv | 162 |
| exp4_probe_linear_olympic_circles.csv | 162 |
| exp4_probe_linear_olympic_pentagons.csv | 162 |
| exp4_probe_linear_subway_connections.csv | 162 |
| exp4_probe_linear_touching_circles.csv | 162 |
| exp4_probe_linear_line_plot.csv | 102 |
| exp4_probe_linear_mmvp.csv | 102 |
| exp4_probe_logreg_circled_letter.csv | 10 |
| exp4_probe_logreg_counting_grid_blank.csv | 10 |
| exp4_probe_logreg_counting_grid_word.csv | 10 |
| exp4_probe_logreg_nested_squares.csv | 10 |
| exp4_probe_logreg_olympic_circles.csv | 10 |
| exp4_probe_logreg_olympic_pentagons.csv | 10 |
| exp4_probe_logreg_subway_connections.csv | 10 |
| exp4_probe_logreg_touching_circles.csv | 10 |
