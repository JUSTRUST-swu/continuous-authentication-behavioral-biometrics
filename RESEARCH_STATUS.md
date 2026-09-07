# 연구 진행 상태

최종 정리: 2026-09-07  
대상: 키스트로크·마우스 행동 생체인증 (KMT / Mendeley, 88 users)  
코드·방법 상세: [DEVELOPMENT.md](DEVELOPMENT.md) · 실행 요약: [README.md](README.md)

---

## 1. 연구 목표

행동 생체특징(dwell / flight / velocity, **6종**)을 **단변량 확률분포**로 모델링하고,  
등록자 분포에 대한 **평균 log-likelihood**로 본인/타인(impostor)을 구분해  
**ROC-AUC / FAR / FRR / EER**을 leakage-free 설정에서 평가한다.

---

## 2. 확정 실험 설계

| 항목 | 설정 |
|------|------|
| Split | `test_N` **6/2/2**, seed **42** |
| Feature set | **all** (6 features) |
| Threshold | **validation_eer** |
| 기본 후보 | Gaussian, Log-normal, Gamma, Weibull, Log-logistic, Student-t |
| GMM | **기본 포함**, K=2 (`--no-include-gmm`로 제외) |
| Ablation | **Local AIC vs Global weighted AIC** (±GMM) |

### AIC 정책

유저 \(u\), feature \(f\), 후보 분포족 \(m\)에 대해 train(유저별 clip+log1p 후)에서 MLE를 적합하고

\[
\mathrm{AIC}_{u,f,m} = 2k_m - 2\,\ell_{u,f,m},
\quad
w_{u,f,m} = n^{\mathrm{used}}_{u,f,m}
\]

(\(k_m\): 자유 파라미터 수, \(\ell\): log-likelihood, \(n^{\mathrm{used}}\): 유효 샘플 수).

- **Local AIC** (등록자 \(e\)):

\[
\hat m^{\mathrm{local}}_{e,f}
= \arg\min_m \mathrm{AIC}_{e,f,m}
\]

- **Global weighted AIC** (코호트 \(U\)):

\[
\overline{\mathrm{AIC}}_{f,m}
= \frac{\sum_{u \in U} w_{u,f,m}\,\mathrm{AIC}_{u,f,m}}{\sum_{u \in U} w_{u,f,m}},
\qquad
\hat m^{\mathrm{global}}_{f}
= \arg\min_m \overline{\mathrm{AIC}}_{f,m}
\]

공유하는 것은 **분포족만**; 등록자 파라미터는 본인 train에서 재추정.  
(`main.py` `best_weighted_mean_aic`와 동일 식.)

---

## 3. 최신 결과 요약 (macro)

숫자는 `comparison_summary.csv`의 `*_macro` 열 (ROC-AUC / FAR / FRR / EER).

### Local vs Global — GMM 없음 (`results/evaluation_aic_selection/`)

| Policy | ROC-AUC | FAR | FRR | EER |
|--------|---------|-----|-----|-----|
| **Local AIC** | **0.843** | 0.190 | 0.341 | 0.183 |
| Global weighted AIC | 0.833 | 0.196 | 0.347 | 0.190 |

- Family agreement: **46.2%** (244 / 528)  
- Local이 소폭 우위.

### Local vs Global — GMM 포함 (`results/evaluation_aic_selection_gmm/`)

| Policy | ROC-AUC | FAR | FRR | EER |
|--------|---------|-----|-----|-----|
| **Local AIC + GMM** | **0.881** | 0.161 | 0.256 | 0.143 |
| Global weighted AIC + GMM | 0.879 | 0.157 | 0.324 | 0.153 |

- Family agreement: **54.7%** (289 / 528)  
- AUC는 거의 동등(local 소폭↑); FRR/EER은 local이 더 낮음.  
- Global(+GMM)은 가중 AIC가 코호트에서 GMM 우세 → family가 GMM으로 쏠림.

### GMM opt-in 참고 (local_aic만)

| Setting | ROC-AUC | FAR | FRR | EER |
|---------|---------|-----|-----|-----|
| Baseline (−GMM) | 0.843 | 0.190 | 0.341 | 0.183 |
| **+ GMM K=2** | **0.881** | 0.161 | 0.256 | 0.143 |

- 성능 향상은 크나 과적합 가능 → 논문에는 **유/무 둘 다** 보고 권장.

---

## 4. 산출물

| 경로 | 내용 |
|------|------|
| `results/evaluation_aic_selection/` | local vs global (−GMM) + ROC 그림 |
| `results/evaluation_aic_selection_gmm/` | local vs global (+GMM) + ROC 그림 |
| `results/evaluation_baseline_no_gmm/` | local_aic −GMM 단독 |
| `results/evaluation_with_gmm/` | local_aic +GMM 단독 |
| `results/main_kmt_gmm/` | 분포 vote (+GMM, train-only) |

그림 재생성:

```bash
python plot_aic_selection_roc_auc.py --root results/evaluation_aic_selection
python plot_aic_selection_roc_auc.py --root results/evaluation_aic_selection_gmm
python plot_model_vote_stacked.py --input-csv results/main_kmt_gmm/tables/model_fit_aggregated_vote_counts.csv --criterion both
```

---

## 5. 다음 단계

- [ ] 논문 Methods에 split / transform / AIC·GMM / threshold 문장 고정  
- [ ] Results 표·그림 캡션 (macro + pooled, agreement)  
- [ ] (선택) GMM 없는 `main.py` vote 산출 (`results/main_kmt/`) 재실행해 ±GMM vote 대비  

---

## 6. 한 줄 요약

> Feature는 **all(6)** 고정. Local vs Global AIC를 GMM 유/무로 비교.  
> −GMM: local ≳ global (0.843 vs 0.833). +GMM: AUC 거의 동등 (0.881 vs 0.879)이나 global family는 GMM으로 쏠림.
