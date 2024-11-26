# quasi-Newton method (Broyden, BFGS)
[2_Newton.md](./2_Newton.md)에서 $\nabla^{2}f(\theta_{k})^-1$를 구하려면 $O(d^{3})$ 시간 복잡도를 가지기도 하고,
$\nabla^{2}$를 구하기 어려운 경우도 있기 때문에 이걸 단순화할 수 없을까?

(아래 부터는 $f(\theta)$의 일 차 도함수를 $g(\theta)$라고 생각하자. 그리고 $\theta$를 $x$라 표기하자!)

접선 대신에 $g(x_{0})$, $g(x_{1})$을 지나는 직선을 사용해 보면 어떨까?
(순간 기울기 대신 평균 기울기를 써보면 어떨까?)

![Secant Method](https://upload.wikimedia.org/wikipedia/commons/9/92/Secant_method.svg)

그러면 $\mathrm{기울기}=\frac{g(x_{k+1})-g(x_{k})}{x_{k+1}-x_{k}}$라고 하고

$s_{k}=x_{k+1} - x_{k}$, $y_{k}=g(x_{k+1})-g(x_{k})$라고 하면 $y_{k}= \mathrm{기울기}\times s_{k}$ ($dg=\mathrm{기울기}\times dx$)

이 때의 $\mathrm{기울기}$를 $B_{k+1}$이라고 하자!

그러면 $y_{k}=B_{k+1}S_{k}$이 되겠다! (secant condition)

만약 $B_{k+1}$를 이전 $B_{k}$, ... $B_{0}$로 표현할 수 있다면?

$$ B_{k+1}=B_{k} + A $$

이렇게 표현해볼 수 있다! 이제 $A$만 구하면 된다!

Broyden은 $B_{k+1} - B_{k}$의 원소들이 매우 작을거라고 생각했다! (작은 learning rate로 안정적이게 수렴해야 하니까)

그래서 아래와 같이 문제를 정의했다!

$$ \min{\parallel B_{k+1}-B_{k}\parallel^{2}_{F}}$$
$$ \text{subject to } y_{k}=B_{k+1}s_{k}$$

즉 secant condition을 만족하면서 $\parallel B_{k+1}-B_{k}\parallel^{2}_{F}$을 최소화 하는 $A$를 찾아보자!

> $\parallel B_{k+1}-B_{k} \parallel^{2}_{F}$가 뭔데?
> 
> Frobenius norm이라는 것으로 모든 원소의 제곱합에 루트를 씌운 것!
> 
> $ \begin{bmatrix}a_{11} && a_{12} \\ a_{21} && a_{22}\end{bmatrix}$의 Frobenius norm은 $\sqrt{a_{11}^{2}+a_{12}^{2}+a_{21}^{2}+a_{22}^{2}}$

이렇게 제약 조건이 있는 문제를 풀 때, Lagrange multiplier 사용하면 쉽게 풀 수 있다.

> Lagrange multiplier는 뭔데?
> 
> 예시로 이해해보자!
> 
> $$\min({x^{2}+y^{2}})$$
> $$ \text{subject to } x+y+2=0$$
> 문제를 풀 때 Lagrange multiplier를 이용하면
> 
> $\mathcal{L}(\lambda, x, y)=x^{2}+y^{2}+\lambda(x+y+2)$
> 
> $\frac{\partial \mathcal{L}}{\partial x}=2x+\lambda=0$
> 
> $\frac{\partial \mathcal{L}}{\partial y}=2y+\lambda=0$
> 
> $\frac{\partial \mathcal{L}}{\partial \lambda}=x+y+2=0$
>  
> 이런 방식으로 쉽게 풀 수 있다!

Lagrange multiplier 이용해서 풀기!

$$\mathcal{L}(\lambda,B_{k+1})=\mathrm{tr}((B_{k+1}-B_{k})^{T}(B_{k+1}-B_{k}))+\lambda^{T}(y_{k}-B_{k+1}s_{k})$$
$$ \frac{\partial \mathcal{L}}{\partial B_{k+1}}=2(B_{k+1}-B_{k})-\lambda s_{k}^{T}=0 $$
정리하면
$B_{k+1}=B_{k}+\frac{1}{2}\lambda s_{k}^{T}$가 된다.

> $$ \frac{\partial }{\partial B_{k+1}}\lambda^{T}(y_{k}-B_{k+1}s_{k})=-\frac{\partial}{\partial B_{k+1}}(\lambda^{T}B_{k+1}s_{k}) $$
> $$ \frac{\partial}{\partial B_{k+1}}(\lambda^{T}B_{k+1}s_{k})=\lambda s_{k}^{T} $$
> 위와 같은 결과가 나오는 이유는 행렬의 곱셈 미분 규칙을 적용할 결과이다!
> 
> $\lambda^{T}B_{k+1}s_{k}$는 스칼라 값인데, $B_{k+1}$에 대해 미분하면 $\lambda^{T}$와 $s_{k}$의 순서를 바꿔서 $\lambda s_{k}^{T}$로 표현된다!

$$ \frac{\partial \mathcal{L}}{\partial \lambda}=y_{k}-B_{k+1}s_{k}=0 $$

$$ y_{k}=B_{k+1}s_{k} $$
$$ y_{k}=(B_{k}+\frac{1}{2}\lambda s_{k}^{T})s_{k} $$
$$ \lambda=2\frac{y_{k}-B_{k}s_{k}}{s_{k}^{T}s_{k}}$$
이렇게 구한 $\lambda$를 $B_{k+1}=B_{k}+\frac{1}{2}\lambda s_{k}^{T}$에 대입하면
$$ B_{k+1}=B_{k}+\frac{1}{2}(2\frac{y_{k}-B_{k}s_{k}}{s_{k}^{T}s_{k}})s_{k}^{T}$$
$$ B_{k+1}=B_{k}+\frac{(y_{k}-B_{k}s_{k})s_{k}^{T}}{s_{k}^{T}s_{k}}$$

짠!

최종 식에서 $y_{k}-B_{k}s_{k}$는 $B_{k}$가 $s_{k}$ 방향에서 $y_{k}$를 얼마나 잘 근사하고 있는지 측정하는 잔차 역할을 하고,

$\frac{(y_{k}-B_{k}s_{k})s_{k}^{T}}{s_{k}^{T}s_{k}}$는 잔차를 보정하여 $B_{k}$를 업데이트 하는 항으로 해석할 수 있다!

근데 여기서 끝나면 Broyden의 논문이 그렇게 까지는 유명해지지 않았을거다.

왜냐면 이렇게 구한 $B_{k+1}$의 inverse를 구해야 하는데 $d\text{ by }d$ Matrix의 inverse를 구하는 건 $O(d^{3})$ 시간 복잡도를 가지기 때문이다!

그래서 애초에 $B_{k+1} = B_{k}+A$가 아니라

$B_{k+1}^{-1}=B_{k}^{-1}+A$로 했으면 되는거 아닌가?

이건 간단하게 [Sherman-Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)로 해결 할 수 있다!

> $(A+uv^{T})^{-1}=A^{-1}-\frac{A^{-1}uv^{T}A^{-1}}{1+v^{T}A^{-1}u}$

$u$를 $y_{k}-B_{k}s_{k}$라고 하고 $v$를 $\frac{s_{k}^{T}}{s_{k}^{T}s_{k}}$라고 한 다음 Sherman-Morrison formula로 표현하고 ... 정리하면

$$ B_{k+1}^{-1}=B_{k}^{-1}+(s_{k}-B_{k}^{-1}y_{k})\frac{s_{k}^{T}B_{k}^{-1}}{s_{k}^{T}B_{k}^{-1}y_{k}}$$

이제 inverse 표시가 필요 없어지고 $B^{-1}$ 대신 $H$로 표현하면
$$ H_{k+1}=H_{k}+(s_{k}-H_{k}y_{k})\frac{s_{k}^{T}H_{k}}{s_{k}^{T}H_{k}y_{k}}$$

드디어 Broyden method 끝!

여기서 $B_{k}$에 더해지는 행렬과 $H_{k}$에 더해지는 행렬은 모두 rank one matrix이다! (vector 외적이기 때문)

따라서 이를 rank one update라고 한다! (왜 rank one update가 중요한가?)

---

근데 Broyden method에는 큰 문제점이 있다. Hessian을 근사하는 것이라고 했는데 Hessian은 항상 symmetric 해야 한다.

$B_{0}$(또는 $H_{0}$ symmetric한 행렬로 설정해도 rank one matrix는 symmetric을 보장하지 않는다!(초기 값을 numerical 하게 계산해서 사용하기도 하지만 Identity 행렬으로 설정하는 경우도 있다.)

BFGS는 Hessian 근사의 정확도를 높이기 위해서 제약 조건에 $A=A^{T}$를 추가했다! (그리고 근사한 Hessian이 positive-definite 일 수 있도록..., Convex optimization에서는 매우 중요!)

$$ \min{\parallel B_{k+1}-B_{k}\parallel^{2}_{W}}$$
$$ \text{subject to } y_{k}=B_{k+1}s_{k} \text{ and } A=A^{T}$$

> $\parallel B_{k+1}-B_{k}\parallel^{2}_{W}$는 weighted frobenius norm이라는 것으로...

최종 식은

$$ B_{k+1}=B_{k}+\frac{y_{k}y_{k}^{T}}{y_{k}^{T}s_{k}}-\frac{B_{k}s_{k}(B_{k}s_{k})^{T}}{s_{k}^{T}B_{k}s_{k}}$$
$$ H_{k+1}=(I-\frac{s_{k}y_{k}^{T}}{y_{k}^{T}s_{k}})H_{k}(I-\frac{y_{k}s_{k}^{T}}{y_{k}^{T}s_{k}})+\frac{s_{k}s_{k}^{T}}{y_{k}^{T}s_{k}}$$

BFGS는 Broyden 방법과 달리 Rank 2 업데이트 이다!

업데이트 항은 Rank 1 행렬인 $\frac{y_{k}y_{k}^{T}}{y_{k}^{T}s_{k}}$와 $\frac{B_{k}s_{k}s_{k}^{T}B_{k}}{s_{k}^{T}B_{k}s_{k}}$로 구성되며, Rank 2 형태가 된다.

정리하자면 Rank 2 업데이트를 하고 대칭성과 positive-definite를 유지하게 하여 Hessian 근사 정확도를 높였으며, 시간 복잡도는 Broyden method와 동일하게 $O(d^{2})$이다.

근데 BFGS는 어디에 좋을까? (SGD는 아직 안배웠지만...)

| **방법**  | **시간 복잡도**   | **장점**                                         | **단점**                                   | **적합한 상황**                                  |
|-----------|--------------|------------------------------------------------|-------------------------------------------|------------------------------------------------|
| GD        | $( O(d^2) )$ | 안정적, 단순                                  | 데이터 크기가 크면 비효율적               | 데이터 작음, 높은 정확도 필요                   |
| SGD       | $( O(d) )$   | 빠르고 데이터 크기에 독립적                   | 수렴 불안정, noisy updates                | 대규모 데이터, 실시간 학습, 비선형 문제         |
| BFGS      | $( O(d^2) )$ | 빠른 수렴, 높은 정확도                        | 고차원 문제에서 메모리 및 계산 부담 큼    | 데이터와 \( d \) 크기 작고, 높은 정확도 필요     |

MSE를 최소하하는 것으로 Objective Function을 설정했을 때는 Hessian이 일정하므로 BFGS의 장점인 Hessian 근사 업데이트가 불필요하다.

그렇지만 Objective function이 복잡한 경우 (특히 비선형이고 Hessian이 변화하는 문제)에서는 BFGS가 SGD보다 유리할 수 있다! (SGD는 nosiy한 업데이트를 수행해서 수렴 경로가 불안정할 수 있다!)

그리고 SGD의 시간 복잡도보다 BFGS의 시간 복잡도가 크지만, 시간 복잡도는 한 번의 iteration에서 나올 수 있는 최대의 경우의 수 이기 때문에 BFGS는 작은 iteration 만으로도 빠르게 수렴할 수 있어 SGD 보다 느리다고 할 수는 없다!
