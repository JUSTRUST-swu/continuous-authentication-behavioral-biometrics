# RnE — Keystroke / Mouse Behavioral Authentication

KMT(Mendeley) 기반 키스트로크·마우스 생체인증 RnE.  
윈도우 feature → 단변량 분포 MLE → held-out 인증 성능(ROC-AUC / FAR / FRR / EER) 평가.

**데이터셋:** https://data.mendeley.com/datasets/fnf8b85kr6/1  
**연구 진행 상태:** [RESEARCH_STATUS.md](RESEARCH_STATUS.md)  
**상세 방법·코드 상태:** [DEVELOPMENT.md](DEVELOPMENT.md)  
**온라인 API 명세:** [API_SPEC.md](API_SPEC.md)

---

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

`scikit-learn`은 GMM(기본 포함)에 필요 (requirements에 포함). 끄려면 `--no-include-gmm`.

---

## Features (6)

| Group | Features |
|-------|----------|
| Dwell / flight | `dwell_mean/std`, `flight_mean/std` |
| Velocity | `velocity_mean/std` |

논문 primary는 **위 6개 전부** (`--feature-set all`).  
(선택 CLI: `dwell` / `flight` / `velocity` — 논문 본평가에는 쓰지 않음.)

- **dwell**: 키 누름 지속시간  
- **flight**: 연속 keydown 간격  
- **velocity**: 마우스 이동 속력  

---

## 논문용 실행 (primary)

```bash
# raw JSON 파싱 시간을 줄이려면 최초 1회 캐시 생성
python -m rne.preprocess

# 인증 평가 (기본: validation EER, local AIC, 6분포 + GMM)
python -m rne.compare --mode authentication_eval

# GMM 제외 / 성분 수 변경
python -m rne.compare --mode authentication_eval --no-include-gmm
python -m rne.compare --mode authentication_eval --gmm-n-components 3

# local AIC vs global mean AIC (동일 split seed; 기본 GMM·비가중)
python -m rne.ablation
# → results/evaluation_aic_selection_gmm_not_weighted/
python -m rne.ablation --weight-global-aic
# → results/evaluation_aic_selection_gmm/
python -m rne.ablation --no-include-gmm
# → results/evaluation_aic_selection/
```

| 산출 | 경로 |
|------|------|
| 단일 auth eval | `results/evaluation/` (기본) |
| AIC ablation (+GMM, 비가중) | `results/evaluation_aic_selection_gmm_not_weighted/` |
| AIC ablation (+GMM, 가중) | `results/evaluation_aic_selection_gmm/` |
| AIC ablation (−GMM) | `results/evaluation_aic_selection/` |

구 threshold(본인 val 하위 5%): `--threshold-mode genuine_quantile`

---

## 논문용 분포 집계 (`main.py`)

인증 평가와 **동일**하게 train 파티션만 사용 (기본 `--fit-split train`, seed 42, 6/2/2).

```bash
python -m rne.modeling --user-range 1 88 --output-dir results/main_kmt
python -m rne.modeling --user-range 1 88 --no-include-gmm --output-dir results/main_kmt_no_gmm
```

주요 산출 (`tables/`):

- `model_fit_aggregated_summary.csv` — `best_weighted_mean_aic` ≈ auth global mean AIC family
- `model_fit_aggregated_vote_counts.csv`  
- `split_assignments.csv`, `model_fit_run_config.json`

`best_weighted_mean_*` 컬럼명은 legacy/API 호환용이다. 기본 실행은 유저별 AIC를 **비가중 평균**하며, `--weight-global-aic`를 줄 때만 `n_used` 가중 평균을 쓴다.

legacy 전 구간 fit: `--fit-split all`

---

## 그림

```bash
python -m rne.plots.aic_selection_roc_auc --root results/evaluation_aic_selection_gmm_not_weighted
python -m rne.plots.aic_selection_roc_auc --root results/evaluation_aic_selection_gmm
python -m rne.plots.aic_selection_roc_auc --root results/evaluation_aic_selection
python -m rne.plots.model_vote_stacked --input-csv results/main_kmt/tables/model_fit_aggregated_vote_counts.csv --criterion both
```

공유 헬퍼: `plotting.py`

---

## 보조 / legacy

```bash
python -m rne.features --user 70
python -m rne.preprocess  # 기본: raw_kmt_dataset → results/preprocessed_kmt
python -m rne.compare --mode train_vs_rest --train-user 1
python -m rne.api_server --host 127.0.0.1 --port 3001
```

온라인 API는 논문 지표용이 아님 (train quantile 근사). 상세: [API_SPEC.md](API_SPEC.md).

---

## 핵심 코드

| 파일 | 역할 |
|------|------|
| `authentication_eval.py` | leakage-free 논문 평가 |
| `loss_compare.py` | CLI (`authentication_eval` + legacy modes) |
| `evaluation_split.py` | train/val/test (`test_N` 우선) |
| `feature_transform.py` | train-only clip + log1p |
| `auth_metrics.py` | FAR/FRR/EER/ROC-AUC |
| `main.py` | 전 user 분포 fit·vote (기본 train-only) |
| `visualize.py` / `preprocess.py` | feature 추출·캐시 |
| `run_aic_selection_ablation.py` | local vs global mean/weighted AIC (±GMM) |
| `plotting.py` | 공유 ROC/score plot 헬퍼 |
| `plot_aic_selection_roc_auc.py` | AIC 정책 ROC / AUC 그림 |
| `plot_model_vote_stacked.py` | 분포 vote stacked bar |
| `api_server.py` | 온라인 train/validate |
| `tests/test_authentication_eval.py` | 단위 테스트 |
