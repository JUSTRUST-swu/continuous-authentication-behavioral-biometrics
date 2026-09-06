# RnE — Keystroke / Mouse Behavioral Authentication

KMT(Mendeley) 기반 키스트로크·마우스 생체인증 RnE.  
윈도우 feature → 단변량 분포 MLE → held-out 인증 성능(ROC-AUC / FAR / FRR / EER) 평가.

**데이터셋:** https://data.mendeley.com/datasets/fnf8b85kr6/1  
**연구 진행 상태:** [RESEARCH_STATUS.md](RESEARCH_STATUS.md)  
**상세 방법·코드 상태:** [DEVELOPMENT.md](DEVELOPMENT.md)  
**온라인 API 명세:** [API_SPEC.md](API_SPEC.md)

---

## Features (6)

| Modality | Features |
|----------|----------|
| Keyboard | `dwell_mean/std`, `flight_mean/std` |
| Mouse | `velocity_mean/std` |
| All | 위 6개 |

- **dwell**: 키 누름 지속시간  
- **flight**: 연속 keydown 간격  
- **velocity**: 마우스 이동 속력  

---

## 논문용 실행 (primary)

```bash
# 인증 평가 (기본: validation EER threshold, local AIC)
python loss_compare.py --mode authentication_eval

# modality ablation (keyboard / mouse / all)
python run_modality_ablation.py

# local AIC vs global weighted AIC
python run_aic_selection_ablation.py

# 결과 그림
python plot_modality_figures.py

# 단위 테스트
pytest -q
```

결과 기본 경로:

- `results/evaluation/`
- `results/evaluation_modality/`
- `results/evaluation_aic_selection/`

구 threshold(본인 val 하위 5%): `--threshold-mode genuine_quantile`

---

## 보조 / legacy

```bash
# 분포 fit 기술 통계
python main.py --user-range 1 88 --output-dir results/main_kmt

# 히스토그램
python visualize.py --user 70

# exploratory risk
python loss_compare.py --mode train_vs_rest --train-user 1

# 온라인 API (논문 지표 아님)
python api_server.py --host 127.0.0.1 --port 3001
```

---

## 핵심 코드

| 파일 | 역할 |
|------|------|
| `authentication_eval.py` | leakage-free 논문 평가 |
| `loss_compare.py` | CLI (`authentication_eval` + legacy modes) |
| `evaluation_split.py` | train/val/test (`test_N` 우선) |
| `feature_transform.py` | train-only clip + log1p |
| `auth_metrics.py` | FAR/FRR/EER/ROC-AUC |
| `main.py` | 전 user 분포 fit·집계 |
| `visualize.py` / `preprocess.py` | feature 추출·캐시 |
| `run_modality_ablation.py` | modality 비교 |
| `run_aic_selection_ablation.py` | local vs global AIC |
| `plot_modality_figures.py` | ROC / EER / FAR·FRR 그림 |
| `api_server.py` | 온라인 train/validate |
