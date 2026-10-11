
### 학습현황

|  | forward | backward | 1→4→2→0→3 | 2→0→4→3→1 | 3→1→0→2→4 |
| --- | --- | --- | --- | --- | --- |
| llava | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-3B | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-8B | ✅ | ✅ | 🔄 3/5 | ❌ | 🔄 3/5 |
| mipha | ✅ | ✅ | ✅ | ✅ | ✅ |
| openflamingo | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen | ✅ | ✅ | ✅ | ✅ | ✅ |

### inference_5task 현황

|  | forward | backward | 1→4→2→0→3 | 2→0→4→3→1 | 3→1→0→2→4 |
| --- | --- | --- | --- | --- | --- |
| llava | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-3B | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-8B | ✅ | ✅ | ❌ | ❌ | ❌ |
| mipha | ✅ | ✅ | ✅ | ✅ | ✅ |
| openflamingo | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen | ✅ | ✅ | ✅ | ✅ | ✅ |

### inference_blind 현황

|  | forward | backward | 1→4→2→0→3 | 2→0→4→3→1 | 3→1→0→2→4 |
| --- | --- | --- | --- | --- | --- |
| llava | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-3B | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-8B | ✅ | ✅ | ❌ | ❌ | ❌ |
| mipha | ✅ | ✅ | ✅ | ✅ | ✅ |
| openflamingo | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen | ✅ | ✅ | ✅ | ✅ | ✅ |

### inference_mmvp+line_plot 현황 (blind 폴더 또는 result_origin)

|  | forward | backward | 1→4→2→0→3 | 2→0→4→3→1 | 3→1→0→2→4 |
| --- | --- | --- | --- | --- | --- |
| llava | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-3B | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-8B | ✅ | ✅ | ❌ | ❌ | ❌ |
| mipha | ✅ | ✅ | ✅ | ✅ | ✅ |
| openflamingo | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen | ✅ | ✅ | ✅ | ✅ | ✅ |

### probing 현황 (inf_clf×3 + inf_reg×3 + blind ben_clf×3 = 9)

|  | forward | backward | 1→4→2→0→3 | 2→0→4→3→1 | 3→1→0→2→4 |
| --- | --- | --- | --- | --- | --- |
| llava | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-3B | ✅ | ✅ | ✅ | ✅ | ✅ |
| vila-8B | ✅ | ✅ | ❌ | ❌ | ❌ |
| mipha | ✅ | ✅ | ✅ | ✅ | ✅ |
| openflamingo | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen | ✅ | ✅ | ✅ | ✅ | ✅ |
