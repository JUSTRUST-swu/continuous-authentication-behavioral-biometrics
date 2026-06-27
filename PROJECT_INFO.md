# Project Info (Current Code 기준)

이 문서는 현재 코드(`main.py`, `visualize.py`, `preprocess.py`, `loss_compare.py`)에서 실제로 사용하는 분석 설정을 정리한 문서입니다.

## 1) 입력 데이터 형식

현재 로더는 아래 두 형식을 모두 지원합니다.

- 신규 세션 로그(flat)
  - 루트: `key_events`, `mouse_events`
  - 선택: `session.monitor_width`, `session.monitor_height`
- 기존 raw_kmt 형식
  - 루트: `true_data` 아래 test 단위
  - 각 test: `key_events`, `mouse_events`

### 실제 사용 필드

- `key_events`: `Event`, `Epoch`, `Key`
- `mouse_events`: `Event`, `Epoch`, `Coordinates`
- `session`: `monitor_width`, `monitor_height` (있으면 좌표 정규화)

## 2) 마우스 좌표 처리

- `monitor_width`, `monitor_height`가 유효하면 좌표를 정규화:
  - `x_norm = x / monitor_width`
  - `y_norm = y / monitor_height`
- 모니터 크기 정보가 없으면 원본 좌표(px)를 그대로 사용합니다.

## 3) 시계열 분할/윈도우 규칙

- gap 분할 기준
  - `0~1s`: normal interval
  - `1~10s`: pause feature
  - `>=10s`: sequence break (세그먼트 분리)
  - `>=30s`: new session break (세그먼트 분리)
- 윈도우 기본값
  - `window_size = 5.0s`
  - `stride = 1.0s`

## 4) 현재 사용하는 Feature

총 6개:

- `dwell_mean`
- `dwell_std`
- `flight_mean`
- `flight_std`
- `velocity_mean`
- `velocity_std`

## 5) 전처리(Feature transform)

각 feature 컬럼에 대해:

1. 유효값 기준 1~99 percentile clipping
2. `log1p` 적용
3. `<= -1` 값은 `NaN` 처리 후 제외

## 6) 통계 모델 후보

현재 fitting 후보 모델(5개):

- Gaussian
- Log-normal
- Gamma
- Weibull
- Student-t

## 7) 모델 선택/평가 기준

### per-user fit 결과

각 `(user, feature, model)`에 대해 저장:

- `log_likelihood`
- `aic`
- `bic`
- `params`
- `n_used`

### feature별 최종 모델 집계

아래 기준을 함께 계산:

1. `majority_vote_aic`
2. `majority_vote_bic`
3. `weighted_mean_aic` (가중치=`n_used`)
4. `weighted_mean_bic` (가중치=`n_used`)
5. `sum_log_likelihood` 최대 모델

## 8) 전처리 JSON 사용 방식

- `preprocess.py`가 세그먼트 단위 시계열 JSON을 생성
- `main.py` / `loss_compare.py`는 기본적으로 preprocessed 파일이 있으면 우선 사용
- 없으면 원본 로그를 직접 읽어 동일 로직으로 계산

## 9) 기본 실행 흐름

1. 전처리 파일 생성
   - `python preprocess.py --dataset-dir ./logs --dataset-pattern \"*.json\" --output-dir results/preprocessed_logs`
2. 전체 사용자 평가
   - `python main.py`
3. loss 비교(선택)
   - `python loss_compare.py --train-user 1 --eval-user-range 2 88`

