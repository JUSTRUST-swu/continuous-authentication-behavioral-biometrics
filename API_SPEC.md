## Keystroke Auth API Spec

프론트엔드에서 바로 붙이기 쉽게 정리한 Markdown 명세입니다.

---

### Base URL
- `http://localhost:3001`

### Content Type
- Request: `application/json`
- Response: `application/json`

---

## 1) Health Check

### `GET /api/health`

#### Response `200`
```json
{
  "status": "ok",
  "model_exists": true,
  "model_info": {
    "updated_at": "2026-08-08T01:23:45+00:00",
    "n_training_sessions": 12
  },
  "n_train_logs": 12
}
```

---

## 2) Session API (핵심)

### `POST /api/session`

`dataType`에 따라 동작이 바뀝니다.

- `train`: 학습 로그 누적 + 모델 재학습
- `validate`: 현재 모델 기준 loss/risk 계산
- `clear`: 학습 상태 초기화

---

### Request Body

```json
{
  "sessionId": "optional-string",
  "dataType": "train | validate | clear",
  "criterionCol": "optional, default=best_weighted_mean_aic",
  "log": {
    "session": {},
    "key_events": [],
    "mouse_events": []
  }
}
```

#### Field Rules
- `sessionId`: 선택값 (없으면 서버가 파일명 자동 생성)
- `dataType`: 필수 (`train` / `validate` / `clear`)
- `log`:
  - `train`, `validate`에서는 필수
  - `clear`에서는 생략 가능
- `criterionCol` (선택):
  - `best_majority_vote_aic`
  - `best_majority_vote_bic`
  - `best_weighted_mean_aic` (기본)
  - `best_weighted_mean_bic`
  - `best_sum_log_likelihood`

---

### 2-1) Train (`dataType=train`)

#### Response `200`
```json
{
  "sessionId": "session-abc",
  "dataType": "train",
  "savedPath": "logs/train/session-abc.json",
  "nTrainingSessions": 5,
  "nTrainingRows": 1234,
  "criterionCol": "best_weighted_mean_aic",
  "features": {
    "dwell_mean": {
      "model": "Log-normal",
      "params": { "shape": 0.07, "loc": 0.0, "scale": 0.09 },
      "n_used": 812
    }
  },
  "baseline": {
    "mean_log_likelihood": 1.45,
    "mean_nll": -1.45
  }
}
```

---

### 2-2) Validate (`dataType=validate`)

#### Response `200`
```json
{
  "sessionId": "session-xyz",
  "dataType": "validate",
  "savedPath": "logs/validate/session-xyz.json",
  "features": {
    "dwell_mean": {
      "model": "Log-normal",
      "n_used": 150,
      "mean_log_likelihood": 1.2,
      "total_log_likelihood": 180.0,
      "mean_nll": -1.2,
      "total_nll": -180.0
    }
  },
  "loss": {
    "mean_log_likelihood": 1.1,
    "mean_nll": -1.1,
    "baseline_mean_log_likelihood": 1.45,
    "ll_diff_vs_train": -0.35,
    "risk_score": 0.35
  }
}
```

#### Risk Score Rule
- `ll_diff_vs_train = mean_log_likelihood - baseline_mean_log_likelihood`
- `risk_score = max(0, -ll_diff_vs_train)`

---

### 2-3) Clear (`dataType=clear`)

#### Response `200`
```json
{
  "dataType": "clear",
  "status": "cleared",
  "removedTrainLogs": 5,
  "removedModelState": true
}
```

---

## 3) Clear 전용 API (대안)

### `POST /api/clear`

Request body 없음.

#### Response `200`
```json
{
  "status": "cleared",
  "removedTrainLogs": 5,
  "removedModelState": true
}
```

---

## Error Responses

### `400 Bad Request`
```json
{
  "error": "dataType must be one of: train, validate, clear"
}
```

### `409 Conflict`
(학습 모델 없이 validate 호출)
```json
{
  "error": "No trained model state found"
}
```

### `500 Internal Server Error`
```json
{
  "error": "Internal error: ..."
}
```

---

## Frontend Fetch Examples

### Train
```javascript
await fetch("http://localhost:3001/api/session", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    dataType: "train",
    log: payload
  })
});
```

### Validate
```javascript
await fetch("http://localhost:3001/api/session", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    dataType: "validate",
    log: payload
  })
});
```

### Clear
```javascript
await fetch("http://localhost:3001/api/session", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    dataType: "clear"
  })
});
```
