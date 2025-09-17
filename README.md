# 전기차 가격 예측 해커톤: 데이터로 EV를 읽다!

> **주최/주관:** 데이콘  
> **대회 주제:** 전기차와 관련된 데이터를 활용하여 전기차 가격을 예측하는 AI 알고리즘 개발  

![포스터](https://api.linkareer.com/attachments/468624)
---

## 배경
전기차(EV, Electric Vehicle)는 전 세계적으로 빠르게 성장하는 시장입니다.  
그러나 차량 가격은 **소비자 구매 의사 결정, 제조사 경쟁 전략, 생산 및 유통 최적화**에 직결되는 중요한 요인입니다.  

따라서 본 프로젝트에서는 **전기차 가격 예측 모델**을 구축하여,  
데이터 기반으로 EV 시장 분석 및 전략 수립에 기여하는 것을 목표로 합니다.

---

## 목표
- **EDA**를 통해 전기차 가격에 영향을 주는 주요 변수를 분석  
- **머신러닝(XGBoost 회귀)** 기반 가격 예측 모델 구축  
- **하이퍼파라미터 튜닝**을 통해 최적 성능 달성  
- 예측 결과를 시각화하고, 실제 가격과 비교  

---

## 데이터셋 개요
대회에서 제공된 **train/test 데이터**는 전기차의 다양한 특성을 포함합니다.

- 주요 변수:
  - `brand` (브랜드명)
  - `year` (출시 연도)
  - `mileage` (주행거리)
  - `battery_capacity` (배터리 용량)
  - `charge_type` (충전 방식)
  - `max_range` (최대 주행 가능 거리)
  - `efficiency` (연비/효율)
  - `motor_power` (모터 출력)
  - `price` (목표 변수, 전기차 가격)

---

## 분석 및 모델링 과정

### 1. EDA (탐색적 데이터 분석)
- 변수별 분포 시각화 (`matplotlib`, `seaborn`)  
- 범주형 변수(`brand`, `charge_type`)와 가격의 관계 분석  
- 수치형 변수(`battery_capacity`, `mileage`, `max_range`)의 상관계수 확인  

**인사이트:**  
- 배터리 용량, 최대 주행거리, 출시 연도가 가격과 강한 양의 상관관계를 가짐  
- 주행거리가 많을수록 가격이 낮아지는 경향  

---

### 2. 데이터 전처리
- 결측치 처리: 평균/최빈값 대체  
- 범주형 변수: **Label Encoding** 및 **One-Hot Encoding** 적용  
- 이상치 제거: 주행거리 극단값 제거  

---

### 3. 모델링
- 사용 알고리즘: **XGBoost Regressor**  
- 하이퍼파라미터 튜닝:  
  ```python
  xgb.XGBRegressor(
      n_estimators=1000,
      learning_rate=0.05,
      max_depth=7,
      subsample=0.8,
      colsample_bytree=0.8,
      reg_alpha=0.1,
      reg_lambda=1.0
  )

### 학습 전략
- Train/Validation Split (80:20)
- Early Stopping 적용 (overfitting 방지)
- RMSE 기준으로 성능 평가

---

### 4. 성능 평가
- 평가 지표: **RMSE (Root Mean Squared Error)**
- 최종 RMSE: **약 0.21 (정규화 스케일)**

**Feature Importance (상위 변수):**
1. 배터리 용량 (`battery_capacity`)
2. 최대 주행거리 (`max_range`)
3. 출시 연도 (`year`)
4. 브랜드 (`brand`)
5. 주행거리 (`mileage`)

---

### 5. 결과 시각화
- **실제 가격 vs 예측 가격** 산점도
- **Feature Importance 그래프** (XGBoost 내장 `plot_importance` 활용)

---

## 결론 및 기대효과
- 전기차 가격은 **배터리·주행거리·출시연도**가 가장 큰 영향을 미침  
- **XGBoost 기반 회귀 모델**로 효율적인 가격 예측 가능성 확인  
- 향후 **딥러닝 모델 (TabNet, CatBoost 등)** 또는 **시계열 수요 예측**으로 확장 가능  
