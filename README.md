# linearADP

scalar.py

satellite tracking antenna example

$J\ddot{\theta}+B\dot{\theta}=T_c$

Reference

[1] https://apmonitor.com/pdc/index.php/Main/ProportionalIntegralDerivative

[2] https://github.com/m-lundberg/simple-pid.git

## TODO

plot.py 만들어서 input은 모델, 출력 원하는 인덱스정도로 두고 자동으로 결과 제시하는 코드 짜기

Q,R 행렬을 일일이 바꿔주어야 하는데 모델에 합치기

클래스 모델에 A, B, C에 대한 정보를 담아두면 dynamics에 대한 메서드를 통일할 수 있을 것 같다

Control Input에 제한을 두어야 할 것 같다

두 가지 이상의 reference 값을 둘 수는 없는지
