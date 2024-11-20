# Newton's method
Objective function이 $(y-A\theta)^{T}(y-A\theta)$ 일 때,
Grandient descent는 $\theta_{k+1}=\theta_{k}+2A^{T}(y-A\theta_{k})\times\alpha$ 하는 방식이었다!

Newton's method는 기본적으로 해를 구하는 방법이다. (어떤 $f(\theta)$가 $0$을 만나는 $\theta^{*}$를 찾아라!)

![Newton's Method](https://en.wikipedia.org/wiki/File:Methode_newton.png)

$x_{0}$에서 어떻게 $x_{1}$으로 가지?
$x_{0}$의 접선을 이용해보자.
높이는 함수 값($f(x)$ 밑변은 $x_{0} - x_{1}$이니까

$$x_{k+1}=x_{k}-\frac{1}{f^{'}(x_{k})}\times f(x_{k})$$

근데 갑자기 왜 zero finding을 할까?

$f(x)=x^{2}$의 경우 미분 값이 0이 되는 지점을 찾고 싶을 때 어떻게 하지?
$f^{'}(x) = 2x$가 0이 되게 하는 $x$를 찾는다!

Objective function의 미분 값이 0이 되는 지점을 찾는 방식으로 최적 해($\theta$)를 찾을 수 있겠네!

$$\theta_{k+1}=\theta_{k}-\frac{1}{f^{''}(\theta_{k})}\times f^{'}(\theta_{k})$$

> $\frac{\partial f}{\partial \theta^{T}}=-2(y-Ax)^{T}A$를 transpose하면 $-2A^{T}(y-Ax)$
> $\frac{\partial }{\partial \theta^{T}}(\frac{\partial f}{\partial \theta^{T}})=2A^{T}A$는 Hessian

(행렬의 역수는 Inverse 이다!)

그래서 최종 식은

$$\theta_{k+1}=\theta_{k}+(2A^{T}A)^{-1}2A^{T}(y-A\theta_{k})$$

애초에 그냥 objective function을 $\frac{1}{2}(y-A\theta)^{T}(y-A\theta)$로 정의 하면 2는 사라지니까

$$\theta_{k+1}=\theta_{k}+(A^{T}A)^{-1}A^{T}(y-A\theta_{k})$$

근데 수식을 풀어 보니...

$$\theta_{k+1}=\theta_{k}+(A^{T}A)^{-1}A^{T}y - (A^{T}A)^{-1}A^{T}A\theta_{k}$$

인데 $A^{T}A$의 inverse를 구할 수 있다면 $\theta_{k}$가 소거 된다!


$$\theta_{k+1}=(A^{T}A)^{-1}A^{T}y$$

이건 많이 보던 **Least-squares solution** 이다!

다음은 quasi-Newton method 중 제일 유명한 BFGS에 대해서 알아봐야지!
