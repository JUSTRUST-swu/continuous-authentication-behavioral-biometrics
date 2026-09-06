# 연구 진행 상태

최종 정리: 2026-09-06  
대상: 키스트로크·마우스 행동 생체인증 (KMT / Mendeley, 88 users)  
코드·방법 상세: [DEVELOPMENT.md](DEVELOPMENT.md) · 실행 요약: [README.md](README.md)

---

## 1. 연구 목표

행동 생체특징(키보드 dwell/flight + 마우스 velocity)을 **단변량 확률분포**로 모델링하고,  
등록자 분포에 대한 **평균 log-likelihood**로 본인/타인(impostor)을 구분해  
**ROC-AUC / FAR / FRR / EER**을 leakage-free 설정에서 평가한다.

---

## 2. 현재까지 확정된 실험 설계

| 항목 | 현재 결정 |
|------|-----------|
| 데이터 | KMT `raw_kmt_user_*.json`, `true_data`, 88명 × `test_1`…`test_10` |
| Split 단위 | **`test_N` = session** (UUID 없음; gap segment로는 60/20/20 불가) |
| Split 비율 | train / val / test = **6 / 2 / 2**, seed **42** |
| Feature | 5s window / 1s stride → 6종 (keyboard 4 + mouse 2) |
| Transform | **train-only** 1–99% clip + log1p |
| 분포 | Gaussian, Log-normal, Gamma, Weibull, Log-logistic, Student-t (**MLE**) |
| 모델 선택 (기본) | 등록자 train **local AIC** |
| Scoring | feature mean LL → feature 평균 (다른 scoring 없음) |
| Decision threshold (기본) | **validation EER** (val genuine vs 타인 val impostor) |
| Attempt | session 단위 (overlapping window ≠ 독립 시도) |
| Primary CLI | `loss_compare.py --mode authentication_eval` |

이전 exploratory 지표(절대 risk = max(0, LL_train−LL_eval))는 **논문 primary에서 제외**.

---

## 3. 완료된 연구·구현 작업

### 방법론 / 평가 프로토콜

- [x] Feature 정의·추출 파이프라인  
- [x] Leakage-free train/val/test 평가 (`authentication_eval`)  
- [x] Split을 `test_N`으로 고정한 근거 정리 (gap으로는 ~39%만 분할 가능)  
- [x] Train-only transform / fit  
- [x] Threshold: 하위 5% quantile → **validation EER**로 기본 변경 (2026-09-06)  
- [x] Legacy risk / train_vs_rest와 논문 경로 분리  

### 실험 (ablation) 코드

- [x] **Modality**: keyboard vs mouse vs all (`run_modality_ablation.py`)  
- [x] **AIC 정책**: local vs global weighted (`run_aic_selection_ablation.py`)  
- [x] 결과 그림: ROC, FAR/FRR–threshold, FAR/FRR 막대, AUC/EER 막대 (`plot_modality_figures.py`)  

### 문서

- [x] README / DEVELOPMENT 통합  
- [ ] 논문 Methods 초안 작성 (체크리스트는 DEVELOPMENT §9)

---

## 4. 지금까지 본 핵심 결과 (참고)

아래 modality 수치는 **threshold = genuine_quantile 0.05** 시점에 돌린 결과이다.  
threshold 기본이 **validation_eer**로 바뀐 뒤에는 **재실행 필요** (FAR/FRR·decision 관련 숫자 변경 가능, ROC-AUC는 threshold 무관이라 유사할 가능성 큼).

| Modality | macro ROC-AUC (대략) | 해석 |
|----------|----------------------|------|
| all (keyboard+mouse) | ≈ **0.84** | 결합이 가장 높음 |
| keyboard | ≈ **0.81** | 단독으로도 강한 편 |
| mouse | ≈ **0.67** | 상대적으로 약함 |

AIC 정책 비교는 스모크(유저 1–3)만 확인됨 → **전체 88명 full run 미완/재확인 필요**.

워크스페이스에 `results/` 산출물이 없으면 위 표는 과거 실행 기준 메모이며, 논문용으로는 다시 돌려 CSV를 고정할 것.

---

## 5. 지금 당장 해야 할 실험 (우선순위)

1. **Threshold=EER 기준으로 본평가 재실행**  
   ```bash
   python loss_compare.py --mode authentication_eval
   python run_modality_ablation.py
   python plot_modality_figures.py
   ```
2. **Local vs Global AIC full run**  
   ```bash
   python run_aic_selection_ablation.py
   ```
3. 결과표·그림 확정 후 논문 Methods / Results에 숫자 고정  

---

## 6. 후속 연구 후보 (미착수 또는 부분)

| 주제 | 상태 |
|------|------|
| Observation duration ablation (5/10/30s…) | 코드 인자만 연결, 실험 러너·결과 없음 |
| dwell / flight 단독 ablation | CLI `feature-set`만 있음 |
| Enrollment quality gate | 미구현 |
| 등록 품질·넓은 분산 → 낮은 식별력 논의 | 정성 관찰만 (예: user 70) |
| 온라인 API를 논문 평가와 동일 프로토콜로 | 미정렬 (API는 근사) |

---

## 7. 해석·보고 시 주의

- Val/test attempt가 user당 보통 **2개** → per-user FAR/FRR·EER 분산 큼 → **macro + pooled** 함께 보고.  
- Decision T(validation EER)와 **test reporting EER**은 다른 값일 수 있음.  
- Global AIC는 test leakage는 아니나, 분포 **족**을 코호트 train으로 공유함을 Methods에 명시.  
- 그림 기본 aggregation은 **pooled** (ROC 곡선과 맞춤); macro 막대는 `--aggregation macro`.

---

## 8. 한 줄 요약

> 프로토콜(leakage-free, `test_N` split, LL scoring, validation EER threshold, local AIC)은 코드상 정착했고, modality 우위(all ≳ keyboard ≫ mouse)는 이전 실행에서 관측됨.  
> **EER threshold 변경 이후 full 재평가 + AIC ablation full run**이 현재 연구의 다음 단계이다.
