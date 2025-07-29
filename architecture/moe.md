## what is the GroupGemm

1. triton tutorial
2. implementation in torchtitan


대부분 groupgemm으로 실행 됨.

satruate gpu via multiple (potentially small) problem

support different gemm shapes and layouts in the same group

no need to manage multi-streaming

a single kernel launch

https://www.youtube.com/watch?v=_rrhYbvNIx0&ab_channel=Triton


## routing type

## loss free, auxilary loss

## ep?

결론 1: GroupGemm과 DenseGemm의 계산 효율성은 입력 크기에 따라 달라지며, 각각의 장점이 존재하는 구간이 있다.
Figure 5(a)에 나타난 것처럼, GroupGemm과 DenseGemm의 계산 효율성은 입력 m이 증가함에 따라 점진적으로 향상된다. 또한 m ≥ 4096일 경우, 동일한 연산량 조건에서 GroupGemm의 계산 효율성은 DenseGemm보다 낮아진다. 반면 m < 2048인 경우에는 GroupGemm의 효율성이 더 높다. 이는 입력 크기에 따라 최적의 커널 구현이 달라지는 것을 의미하며, 이를 문제 형태에 따른 커널 튜닝이라 한다.

MoE 모델의 추론 과정에서는 입력 부하가 자주 변한다. 예를 들어, prefill 단계에서는 입력이 주로 DenseGemm이 더 빠른 구간에 해당하며, decode 단계에서는 GroupGemm이 더 빠른 구간에 해당한다.

결론 2: GroupGemm의 경우, 입력 크기가 일정 수준 이상이 되면 group 수를 늘려도 처리량이 향상되지 않는다.
Group 수를 증가시키는 것은 특히 미세한(Minute-grained) MoE 모델에서 계산 처리량을 향상시키는 일반적인 방법이었다. 이 방법은 입력 크기가 작을 때 효과적이지만, 입력 크기가 크면 group 수를 늘려도 계산 처리량이 더 이상 증가하지 않는다(Figure 5(b) 참조). 즉, 큰 입력 크기에서는 하나의 큰 GroupGemm을 여러 개의 작은 GroupGemm으로 분할해도 처리량 감소가 발생하지 않는다.

결론 3: GroupGemm의 경우, 입력 크기가 일정 수준 이상이 되면 더 많은 SM을 사용해도 처리량이 증가하지 않는다.
Nanoflow [39]는 DenseGemm의 경우, 일정 범위 내에서는 할당된 SM 수를 줄여도 GEMM의 계산 처리량에는 영향을 미치지 않는다고 지적하였다. 우리는 GroupGemm에 대해서도 유사한 경향을 Figure 5(c)에서 확인하였다. 또한, 서로 다른 group 수를 가진 GroupGemm의 테스트 결과, SM 수를 줄여도 실행 효율성에는 영향이 없음을 보여주었다.

MoE 추론 중에는 layernorm, residual, activation, topKGating 등과 같은 메모리 바운드 연산자들이 많이 존재하며, 이들에 대해서는 커널 퓨전(kernel fusion), 가중치 양자화(weight quantization), 벡터화된 메모리 접근(vectorized memory access) [2] 등의 전통적인 최적화 방법이 존재한다. 또 다른 메모리 바운드 연산자는 Attention의 KVCache 로딩으로, 이에 대해서도 많은 논문들이 다루어왔다.

Figure 6에서 보듯이, 이 임계값은 연산자의 계산 부하와 밀접한 관련이 있다. 토큰 수가 64일 경우, 40개의 SM으로도 충분한 커널 성능을 달성할 수 있다. 토큰 수가 128, 256, 혹은 2048일 경우, 60개의 SM으로도 충분하다. 입력 행(row)의 수가 적은 경우에는 MoE GEMM도 메모리 바운드가 되며, 이 경우에도 유사한 경향을 확인할 수 있다.

Figure 7에서 볼 수 있듯이, GPU 수가 많아질수록 통신에 소요되는 시간의 비중도 증가한다. 만약 GroupGemm 커널을 튜닝하여 더 효율적으로 만들면, 상대적으로 통신 시간이 차지하는 비중은 더욱 커지게 된다.

기존의 수평 분할 방식은 파라미터 행렬에 대한 반복적인 메모리 I/O를 유발하는 경향이 있다.