# GD(Gradient Descent)
배치 Gradient Descent라고도 불리며, 모든 데이터 포인트(전체 배치)를 사용해 기울기를 계산합니다.

수식:
GD는 다음과 같은 반복적인 업데이트를 통해 파라미터 𝜃를 최적화합니다.

$\theta_{t+1} = \theta_{t} - \eta\cdot\nabla_{\theta}J(\theta)$

- $\theta_{t}$ : t-번째 단계에서의 파라미터 벡터
- $\eta$: 학습률(Learning Rate)
- $J(\theta)$: 손실 함수(Loss Function)
- $\nabla_{\theta}J(\theta)$: 손실 함수의 **전체 데이터***에 대한 기울기

#### 기울기 계산
$\nabla_{\theta}J(\theta) = {1\over N}\displaystyle\sum_{i=1}^{N}\nabla_{\theta}J_{i}(\theta)$
- $N$: 전체 데이터 포인트 개수
- $J_{i}({\theta})$: i-번째 데이터 샘플에 대한 손실 함수 값


#### 특징
- 장점: 기울기의 정확한 계산으로 안정적인 수렴.
- 단점: 데이터가 많을 경우 계산 비용이 매우 높음.

----

# SGD(Stochastic Gradient Descent)
SGD는 전체 데이터 대신 랜덤으로 선택된 하나의 샘플만 사용해 기울기를 계산합니다. 이는 계산 비용을 크게 줄이지만, 더 많은 진동(Noise)을 포함합니다.

#### 수식:
SGD는 다음과 같은 방식으로 파라미터를 업데이트합니다:

$\theta_{t+1} = \theta_{t} - \eta\cdot\nabla_{\theta}J_{i_{t}}(\theta) $
- $i_t$: t-번째 단계에서 랜덤으로 선택된 데이터 샘플의 인덱스
- $\nabla_{\theta}J_{i_{t}}(\theta)$: 단일 샘플 $i_t$에 대한 손실 함수의 기울기

#### 특징
- 장점: 빠른 업데이트, 잡음을 포함해 Local Minimum을 벗어날 수도 있고, 더 많은 경로를 탐색. 비용 효율적
- 단점: 기울기 계산에 포함된 노이즈로 인해 최적화 과정에서 진동이 발생.
경사가 완만한 영역에서는 수렴 속도가 느려질 수 있음. 
적절한 학습률을 선택하지 않으면 최적화가 잘 되지 않음. 



## 주요 차이점 요약


----
##### 평균, MSE, 