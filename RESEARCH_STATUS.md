# 연구 진행 상태

최종 정리: 2026-09-07 (GMM 포함 local vs global AIC 완료)  
대상: 키스트로크·마우스 행동 생체인증 (KMT / Mendeley, 88 users)  
코드·방법 상세: [DEVELOPMENT.md](DEVELOPMENT.md) · 실행 요약: [README.md](README.md)

---

## 1. 연구 목표

행동 생체특징(키보드 dwell/flight + 마우스 velocity)을 **단변량 확률분포**로 모델링하고,  
등록자 분포에 대한 **평균 log-likelihood**로 본인/타인(impostor)을 구분해  
**ROC-AUC / FAR / FRR / EER**을 leakage-free 설정에서 평가한다.

---

## 2. 확정 실험 설계 (이번 런)


| 항목          | 설정                                                            |
| ----------- | ------------------------------------------------------------- |
| Split       | `test_N` 6/2/2, seed **42**                                   |
| Feature set | **all** (keyboard + mouse)                                    |
| Threshold   | **validation_eer**                                            |
| 기본 후보       | Gaussian, Log-normal, Gamma, Weibull, Log-logistic, Student-t |
| GMM         | `--include-gmm`, K=2 (opt-in 비교)                              |
| Ablation    | **Local AIC vs Global weighted AIC** (±GMM)                   |


**AIC 정책 (가중치):**

유저 \(u\), feature \(f\), 후보 분포족 \(m\)에 대해 train(유저별 clip+log1p 후)에서 MLE를 적합하고

\[
\mathrm{AIC}_{u,f,m} = 2k_m - 2\,\ell_{u,f,m},
\quad
w_{u,f,m} = n^{\mathrm{used}}_{u,f,m}
\]

(\(k_m\): 자유 파라미터 수, \(\ell\): log-likelihood, \(n^{\mathrm{used}}\): 유효 샘플 수).

- **Local AIC** (등록자 \(e\)): feature별로 본인 train만 보고 선택.

\[
\hat m^{\mathrm{local}}_{e,f}
= \arg\min_m \mathrm{AIC}_{e,f,m}
\]

- **Global weighted AIC:** 코호트 \(U\) 전원 train AIC를 \(n^{\mathrm{used}}\)로 가중 평균한 뒤 feature별 공통 분포족 선택.

\[
\overline{\mathrm{AIC}}_{f,m}
= \frac{\sum_{u \in U} w_{u,f,m}\,\mathrm{AIC}_{u,f,m}}{\sum_{u \in U} w_{u,f,m}},
\qquad
\hat m^{\mathrm{global}}_{f}
= \arg\min_m \overline{\mathrm{AIC}}_{f,m}
\]

샘플이 많은 유저가 family 선택에 더 반영됨. **공유하는 것은 분포족 \(\hat m\)만**; 등록자 파라미터는 본인 train에서 재추정. (`main.py`의 `best_weighted_mean_aic`와 동일 식.)

> 키보드/마우스 modality ablation은 실험 항목에서 **제외** (feature는 all 고정).

---

