# 코드 개발 상태 & 실험 방법론

코드 기준 현재 상태, 파이프라인, 데이터 split 근거, 실행·한계를 한곳에 정리한다.

**논문 primary:** `python loss_compare.py --mode authentication_eval`  
**연구 진행 메모:** [RESEARCH_STATUS.md](RESEARCH_STATUS.md)  
**빠른 시작:** [README.md](README.md) · **API:** [API_SPEC.md](API_SPEC.md)

최종 문서 동기화: 2026-09-07 (레이아웃 단순화 반영; keyboard/mouse modality 실험 코드·문서 제거).

---

## 1. 개발 상태 요약

| 영역 | 상태 | 비고 |
|------|------|------|
| Feature 추출 (dwell/flight/velocity) | 완료 | 5s window / 1s stride |
| Preprocessed 캐시 | 완료 | schema v2에 `session_id` 권장; eval은 raw `test_N`으로도 동작 |
| 단변량 6분포 MLE + AIC/BIC | 완료 | Gaussian, Log-normal, Gamma, Weibull, Log-logistic, Student-t |
| GMM (기본 on) | 완료 | 기본 K=2; `--no-include-gmm`로 제외; sklearn |
| `main.py` train-only vote | 완료 | 기본 `--fit-split train` (auth와 동일 split/transform) |
| Leakage-free auth eval | 완료 | train/val/test, train-only transform/fit |
| Decision threshold | 완료 | **기본 = validation EER** (`genuine_quantile` 옵션 유지) |
| Local vs Global AIC ablation | 완료 | `run_aic_selection_ablation.py` (±`--no-include-gmm` / `--weight-global-aic`) |
| 결과 시각화 | 완료 | `plotting.py` + AIC ROC / vote stacked |
| 단위 테스트 | 부분 | `tests/test_authentication_eval.py` |
| 온라인 API | 동작 | 논문 숫자용 아님 (train quantile 근사) |
| Enrollment quality gate | 미구현 | 후속 후보 |
| Observation duration ablation | 미구현 | `--window-size`는 연결됨, 체계적 러너 없음 |
| dwell / flight / velocity 단독 | CLI만 | `--feature-set {all,dwell,flight,velocity}` |

### 구현 완료된 실험 산출물 (예시)

- `results/evaluation_aic_selection/` — local vs global (−GMM)  
- `results/evaluation_aic_selection_gmm/` — local vs global (+GMM)  
- `results/evaluation_baseline_no_gmm/`, `results/evaluation_with_gmm/` — local_aic GMM 유/무 참고  
- `results/main_kmt_gmm/` — 논문용 분포 vote (+GMM, train-only)  

### 의도적으로 legacy로 둔 것

- `train_vs_rest` / `all_vs_rest` / `user_compare` / `validate_risk`  
- 절대 risk `max(0, LL_train − LL_eval)` — 논문 primary 아님  
- Summary 맵에 `GMM`이 있으면 legacy fit 경로가 GMM fitter를 **자동 enable** (조용히 feature 스킵하지 않음)

### 제거된 것 (실험 미사용)

- keyboard vs mouse vs all modality 러너/플롯 (`run_modality_ablation`, `plot_modality_*`)  
- 미사용 유틸 (`compare.py`, `mouse_xy_ranges.py`)  
- `--feature-set keyboard|mouse` alias  

---

## 2. 파이프라인

```text
raw JSON (true_data.test_1 … test_10)
  → (optional) preprocessed_kmt: gap segment + 시계열 캐시
  → sliding window 5s / stride 1s → feature 6종
  → train-only 1–99% clip + log1p
  → 분포 6종 MLE (+ optional GMM)
  → family 선택: local AIC 또는 global weighted AIC
  → [논문] train fit → val EER threshold → test genuine vs impostor
  → [legacy] train_vs_rest 등
```

### 점수

- **Score** = feature별 mean log-likelihood의 평균 (높을수록 본인 같음)  
- genuine / impostor **동일 LL scoring**  
- Accept iff `score >= threshold`

### Threshold (기본)

- `validation_eer`: 등록자 val genuine vs 타인 val impostor에서 FAR≈FRR인 threshold  
- 옵션 `genuine_quantile` (기본 0.05): val genuine만으로 하위 분위수  
- **Test EER**은 보고용; decision T와 별개로 test에서 다시 계산  

### 지표

| 지표 | 의미 |
|------|------|
| ROC-AUC | threshold 무관 |
| FAR / FRR | decision threshold 기준 |
| EER | test에서 FAR≈FRR (보고용) |
| macro | user별 지표 평균 |
| pooled | 전체 attempt 합쳐 계산 |

Attempt 단위 = **session (`test_N`)**. overlapping 5s window ≠ 독립 시도.

---

## 3. 데이터 & Split 근거 (KMT)

Probe: 2026-09-05, `raw_kmt_dataset/raw_kmt_user_*.json` (n=88).

| 수준 | JSON에 있는 것 |
|------|----------------|
| User | `details`, `true_data`, `false_data` |
| Trials | `test_1` … `test_10` (전원 10개) |
| Session UUID | **없음** |

### Gap segment로는 60/20/20 불가

| 단위 | 60/20/20 가능 |
|------|----------------|
| Gap segments | **34/88 (≈39%)** |
| `test_N` sessions | **88/88 (100%)** → e.g. 6/2/2 |

### Split 우선순위 (`evaluation_split.py`)

1. **Primary:** `session_id` = `test_N`  
2. Fallback: gap `segment_id`  
3. Last resort: contiguous time-block  

기본 비율 seed=42, train/val/test = 0.6/0.2/0.2. 분석은 `true_data`.

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

### 후보 분포 (MLE)

| 모델 | 대략 k | 비고 |
|------|--------|------|
| Gaussian | 2 | |
| Log-normal | 2 | floc=0, x>0 |
| Gamma | 2 | floc=0, x>0 |
| Weibull | 2 | floc=0, x>0 |
| Log-logistic | 2 | SciPy `fisk`, floc=0, x>0 |
| Student-t | 3 | |
| GMM (기본) | 3K−1 | 기본 포함; `--no-include-gmm`로 제외; sklearn 1D, default K=2 |

\[
\mathrm{AIC} = 2k - 2\ell,\qquad \mathrm{BIC} = k\ln n - 2\ell
\]

**GMM**은 기본 후보에 포함. `--no-include-gmm`일 때만 제외.

### 모델 선택

수식·기호 정의: [RESEARCH_STATUS.md](RESEARCH_STATUS.md) §2.

| 용도 | 규칙 |
|------|------|
| `authentication_eval` 기본 | 등록자 train **local AIC** |
| AIC ablation | `local_aic` vs `global_weighted_aic` |
| `main.py` vote (**기본**) | `--fit-split train` → majority / weighted AIC·BIC / sum LL |
| `main.py --fit-split all` | legacy 전 구간 descriptive fit |
| Legacy API / `user_compare` | summary의 `best_weighted_mean_aic` (GMM 맵 → 자동 enable) |

`global_weighted_aic`와 `main.py`의 `best_weighted_mean_aic`는 동일 유저·seed·GMM 플래그면 **분포족**이 일치해야 함. 파라미터는 등록자 train에서 재추정.

---

## 5. 핵심 파일 맵

| 파일 | 역할 |
|------|------|
| `visualize.py` | 이벤트 로드, feature, 히스토그램 |
| `preprocess.py` | preprocessed 캐시 |
| `main.py` | 분포 fit·vote (기본 train-only) |
| `authentication_eval.py` | 논문 eval |
| `evaluation_split.py` | split |
| `feature_transform.py` | train-only transform |
| `auth_metrics.py` | FAR/FRR/EER/AUC |
| `loss_compare.py` | CLI 진입점 |
| `run_aic_selection_ablation.py` | AIC 정책 비교 |
| `plotting.py` | 공유 ROC / pooled-score 헬퍼 |
| `plot_aic_selection_roc_auc.py` | AIC 정책 그림 |
| `plot_model_vote_stacked.py` | vote stacked bar |
| `api_server.py` | 온라인 API |
| `data_collection.py` | 로컬 로그 수집 |
| `tests/test_authentication_eval.py` | 단위 테스트 |

---

## 6. 자주 쓰는 명령

```bash
pip install -r requirements.txt

python loss_compare.py --mode authentication_eval
python loss_compare.py --mode authentication_eval --user-range 1 5 --output-dir results/evaluation_smoke

python run_aic_selection_ablation.py
python run_aic_selection_ablation.py --weight-global-aic
python run_aic_selection_ablation.py --no-include-gmm

python plot_aic_selection_roc_auc.py --root results/evaluation_aic_selection_gmm_not_weighted
python plot_model_vote_stacked.py --criterion both

python main.py --user-range 1 88 --output-dir results/main_kmt_gmm_not_weighted
python main.py --user-range 1 88 --no-include-gmm --output-dir results/main_kmt
python main.py --user-range 1 88 --fit-split all --output-dir results/main_kmt_all

python visualize.py --user 70
pytest -q
```

---

## 7. 알려진 한계

- Val session이 보통 2개 → threshold(EER/quantile) 불안정 warning 정상.  
- User당 test attempt 2개 → per-user FAR/FRR 표본 작음.  
- Pooled 지표는 등록자마다 impostor를 다시 넣어 attempt가 O(N²)로 중복될 수 있음 → macro와 함께 제시.  
- API는 train-session quantile 근사; **논문 숫자는 offline `authentication_eval`**.  
- Preprocessed v1(session_id 없음)은 auth에서 raw `test_N` 경로가 안전.  
- `results/`, `logs/`, `raw_kmt_dataset/` 는 보통 gitignore.

---

## 8. 후속 후보

- [ ] Enrollment quality gate  
- [ ] Observation duration ablation 실험 러너 (5/10/30s…)  
- [ ] dwell / flight / velocity 단독 ablation 결과 정리  
- [ ] 테스트 확장 (EER path, end-to-end smoke)  
- [ ] API를 외부에 열 경우 인증 (현재 CORS `*`, auth 없음 — localhost 전제)  
- [ ] 논문 Methods / Results 초안 (숫자 고정)

---

## 9. 논문 Methods 체크리스트

- [x] 데이터셋·사용자 수·`true_data` (KMT, 88)  
- [x] Split = `test_N` 6/2/2, seed 42  
- [x] Window 5s / stride 1s, feature 6종 (`all`)  
- [x] Train-only clip + log1p  
- [x] 6분포 MLE + local AIC; global / ±GMM ablation 실행 완료  
- [x] Score = mean LL; threshold = **validation EER**  
- [x] Test: ROC-AUC, FAR/FRR@T, reporting EER  
- [ ] Methods 문장으로 고정·그림 캡션 정리  
