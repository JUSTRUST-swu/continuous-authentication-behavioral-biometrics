# 코드 개발 상태 & 실험 방법론

코드 기준 현재 상태, 파이프라인, 데이터 split 근거, 실행·한계를 한곳에 정리한다.

**논문 primary:** `python loss_compare.py --mode authentication_eval`  
**연구 진행 메모:** [RESEARCH_STATUS.md](RESEARCH_STATUS.md)  
**빠른 시작:** [README.md](README.md) · **API:** [API_SPEC.md](API_SPEC.md)

---

## 1. 개발 상태 요약

| 영역 | 상태 | 비고 |
|------|------|------|
| Feature 추출 (dwell/flight/velocity) | 완료 | 5s window / 1s stride |
| Preprocessed 캐시 | 완료 | schema v2에 `session_id` 권장; eval은 raw `test_N`으로도 동작 |
| 단변량 6분포 MLE + AIC/BIC | 완료 | Gaussian, Log-normal, Gamma, Weibull, Log-logistic, Student-t |
| Leakage-free auth eval | 완료 | train/val/test, train-only transform/fit |
| Decision threshold | 완료 | **기본 = validation EER** (`genuine_quantile` 옵션 유지) |
| Modality ablation | 완료 | keyboard / mouse / all |
| Local vs Global AIC ablation | 완료 | `run_aic_selection_ablation.py` |
| 결과 시각화 | 완료 | `plot_modality_figures.py` |
| 단위 테스트 | 부분 | `tests/test_authentication_eval.py` |
| 온라인 API | 동작 | 논문 숫자용 아님 (train quantile 근사) |
| Enrollment quality gate | 미구현 | 후속 후보 |
| Observation duration ablation | 미구현 | `--window-size`는 연결됨, 체계적 실험 스크립트는 없음 |
| dwell / flight 단독 ablation | CLI만 | `feature-set` choices에 있음 |

### 구현 완료된 실험 산출물 (예시 경로)

- `results/evaluation/` — 전체 auth eval  
- `results/evaluation_modality/` — modality 비교 + `figures/`  
- `results/evaluation_aic_selection/` — local vs global AIC  
- `results/main_kmt/tables/` — 기술용 분포 fit 집계  

### 의도적으로 legacy로 둔 것

- `train_vs_rest` / `all_vs_rest` / `user_compare` / `validate_risk`  
- 절대 risk `max(0, LL_train − LL_eval)` — 논문 primary 아님  

---

## 2. 파이프라인

```text
raw JSON (true_data.test_1 … test_10)
  → (optional) preprocessed_kmt: gap segment + 시계열 캐시
  → sliding window 5s / stride 1s → feature 6종
  → train-only 1–99% clip + log1p
  → 분포 6종 MLE
  → [논문] train fit → val EER threshold → test genuine vs impostor
  → [legacy] train_vs_rest 등
```

### 점수

- **Score** = feature별 mean log-likelihood의 평균 (높을수록 본인 같음)  
- genuine / impostor **동일 LL scoring** (다른 점수 체계 없음)  
- Accept iff `score >= threshold`

### Threshold (기본)

- `validation_eer`: 등록자 val genuine vs 타인 val impostor에서 FAR≈FRR인 threshold  
- 옵션 `genuine_quantile` (기본 0.05): val genuine만으로 하위 분위수  
- **Test EER**은 보고용(reporting); decision T와 별개로 test에서 다시 계산해 표에 넣음  

### 지표

| 지표 | 의미 |
|------|------|
| ROC-AUC | threshold 무관 |
| FAR / FRR | decision threshold 기준 |
| EER | test에서 FAR≈FRR (보고용) |
| macro | user별 지표 평균 |
| pooled | 전체 attempt 합쳐 계산 |

Attempt 단위 = **session (`test_N`)**. overlapping 5s window를 독립 시도로 쓰지 않음.

---

## 3. 데이터 & Split 근거 (KMT)

Probe: 2026-09-05, `raw_kmt_dataset/raw_kmt_user_*.json` (n=88).

| 수준 | JSON에 있는 것 |
|------|----------------|
| User | `details`, `true_data`, `false_data` |
| Trials | `test_1` … `test_10` (전원 10개) |
| Session UUID | **없음** |

### Gap segment로는 60/20/20 불가

연속 `test_N` 간격이 매우 짧아 gap(≥10s/≥30s)으로 나누면 세그먼트가 부족한 사용자가 많음.

| 단위 | 60/20/20 가능 |
|------|----------------|
| Gap segments | **34/88 (≈39%)** |
| `test_N` sessions | **88/88 (100%)** → e.g. 6/2/2 |

### Split 우선순위 (`evaluation_split.py`)

1. **Primary:** `session_id` = `test_N`  
2. Fallback: gap `segment_id`  
3. Last resort: contiguous time-block  

기본 비율 seed=42, train/val/test = 0.6/0.2/0.2.

분석은 `true_data` 사용. 모니터 크기가 있으면 마우스 좌표를 [0,1] 정규화.

---

## 4. Feature · 전처리 · 분포

### Gap / window

| Δt | 처리 |
|----|------|
| ≤1s | 동일 세그먼트 |
| 1–10s | pause (분리 안 함) |
| ≥10s | sequence break |
| ≥30s | session break |

- Window 5.0s, stride 1.0s (CLI로 변경 가능; auth eval에 전달됨)

### Transform (논문 경로)

Train만으로 1–99% clip bounds 추정 → val/test에 동일 적용 → `log1p`.  
(`feature_transform.py`)

### 후보 분포 (MLE, SciPy)

| 모델 | 대략 k | 비고 |
|------|--------|------|
| Gaussian | 2 | |
| Log-normal | 2 | floc=0, x>0 |
| Gamma | 2 | floc=0, x>0 |
| Weibull | 2 | floc=0, x>0 |
| Log-logistic | 2 | SciPy `fisk`, floc=0, x>0 |
| Student-t | 3 | |

AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ.

### 모델 선택 규칙 (헷갈리기 쉬운 부분)

| 용도 | 규칙 |
|------|------|
| `main.py` 기술 집계 | majority / weighted AIC·BIC / sum LL **전부** 저장 |
| Legacy API / `user_compare` | summary의 `best_weighted_mean_aic` |
| **`authentication_eval` 기본** | 등록자 **train local AIC** |
| AIC ablation | `local_aic` vs `global_weighted_aic` |

`global_weighted_aic`: 코호트 **train** 파티션에서 (유저별 train-only transform 후) 가중 평균 AIC로 **분포족만** 공유. 파라미터는 등록자 train에 재추정. test leakage 아님(집단 정보 공유는 Methods에 명시).

---

## 5. 핵심 파일 맵

| 파일 | 역할 |
|------|------|
| `visualize.py` | 이벤트 로드, feature, 히스토그램 |
| `preprocess.py` | preprocessed 캐시 |
| `main.py` | 전/단일 user 분포 fit·집계 |
| `authentication_eval.py` | 논문 eval |
| `evaluation_split.py` | split |
| `feature_transform.py` | train-only transform |
| `auth_metrics.py` | FAR/FRR/EER/AUC |
| `loss_compare.py` | CLI 진입점 |
| `run_modality_ablation.py` | modality 비교 |
| `run_aic_selection_ablation.py` | AIC 정책 비교 |
| `plot_modality_figures.py` | 그림 (ROC 등은 **pooled** 점수; 막대 기본 pooled) |
| `api_server.py` | 온라인 API |
| `compare.py`, `data_collection.py`, `mouse_xy_ranges.py` | 유틸 |

---

## 6. 자주 쓰는 명령

```bash
# 논문 인증 평가
python loss_compare.py --mode authentication_eval
python loss_compare.py --mode authentication_eval --user-range 1 5 --output-dir results/evaluation_smoke

# modality / AIC ablation (동일 split seed)
python run_modality_ablation.py
python run_aic_selection_ablation.py

# 그림
python plot_modality_figures.py
# 막대만 macro로 보고 싶을 때:
python plot_modality_figures.py --aggregation macro

# 분포 fit (descriptive)
python main.py --user-range 1 88 --output-dir results/main_kmt

# 히스토그램
python visualize.py --user 70

# 테스트
pytest -q
```

---

## 7. 알려진 한계

- Val session이 보통 2개 → threshold(EER/quantile) 불안정 warning 정상.  
- User당 test attempt 2개 → per-user FAR/FRR 표본 작음.  
- Pooled 지표는 등록자마다 impostor를 다시 넣어 attempt가 O(N²)로 중복될 수 있음 → macro와 함께 제시.  
- API는 train-session quantile 근사; **논문 숫자는 offline `authentication_eval`**.  
- Preprocessed v1(session_id 없음)은 auth에서 raw `test_N` 경로를 쓰는 것이 안전.  
- `results/`, `logs/`, `raw_kmt_dataset/` 는 보통 gitignore.

---

## 8. 후속 후보

- [ ] Enrollment quality gate  
- [ ] Observation duration ablation 실험 러너 (5/10/30s…)  
- [ ] dwell / flight 단독 ablation 결과 정리  
- [ ] 테스트 확장 (EER path, end-to-end smoke)  
- [ ] API를 외부에 열 경우 인증 (현재 CORS `*`, auth 없음 — localhost 전제)

---

## 9. 논문 Methods 체크리스트

- [ ] 데이터셋·사용자 수·`true_data`  
- [ ] Split = `test_N` 6/2/2, seed  
- [ ] Window 5s / stride 1s, feature 6종  
- [ ] Train-only clip + log1p  
- [ ] 6분포 MLE, **local AIC** (또는 global ablation 명시)  
- [ ] Score = mean LL; threshold = **validation EER**  
- [ ] Test: ROC-AUC, FAR/FRR@T, reporting EER  
- [ ] Modality / AIC ablation 설정 동일 여부  
