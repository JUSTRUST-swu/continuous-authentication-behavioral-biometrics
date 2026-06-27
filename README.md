# 데이터셋 출처
https://data.mendeley.com/datasets/fnf8b85kr6/1

# 용어 정리
dwell: 입력의 길이
flight: 입력과 입력 사이의 간격
velocity: 움직임의 속도

mean: 평균
std: 표준편차


# Feature 정의

## Keyboard feature
dwell_mean
dwell_std

flight_mean
flight_std

## Mouse feature
velocity_mean
velocity_std


# 통계 모델 적합성 평가 결과

## 통계 모델 후보
Gaussian, Log-normal, Gamma, Weibull, Student-t

## 모델 평가 방식
각각의 user data에 모든 모델을 fit, AIC 및 BIC 평과 결과로 최적 모델에 vote. Vote 취합 후 최적 모델 선정.

## 모델 평가 결과

## JSON 간소화
- 분석에 직접 쓰는 필드만 남긴 JSON 생성:
`python simplify_raw_kmt.py --dataset-dir ./raw_kmt_dataset --output-dir results/simplified_raw_kmt`
- 생성된 간소화 포맷은 `visualize.py` 로더에서 그대로 읽을 수 있음.

# 로그 수집 서버

프론트엔드에서 `:3000`으로 세션 로그 JSON을 보내면 요청 1건당 JSON 파일 1개를 저장합니다.

## 실행
`python log_receiver.py --host 0.0.0.0 --port 3000 --base-dir ./logs`

## 엔드포인트
- `POST /logs`
- `POST /api/logs`

## 저장 구조
`./logs/20260627T145301_123456Z_ab12cd34.json`

## 요청 예시 (fetch)
```javascript
fetch("http://localhost:3000/logs", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
```

## 요청 예시 (curl)
`curl -X POST http://localhost:3000/logs -H "Content-Type: application/json" --data @sample_log.json`
