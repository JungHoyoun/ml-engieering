# 간단한 계산으로 동료들을 감동시키자

## **파라미터 수 계산법**

Dense 모델의 전체 파라미터 수는 다음과 같은 근사식을 통해 빠르게 계산할 수 있습니다:

* **N**: 전체 파라미터 수
* $h$: hidden dimension
* $v$: vocabulary 크기
* $l$: 레이어 수 (각 레이어는 Attention + FFN 포함)
* $r$: FFN 확장 비율 (보통 $4\times$, LLaMA 3.1은 $\tfrac{8}{3} \times 1.3 \approx 3.467$)
* RMSNorm 등 기타 구성 요소는 상대적으로 미미하므로 생략 가능

$$
N = 2\,h\,v \;+\; l \times \bigl(2.5\,h^2 + 3\,rh^2\bigr)
$$

위 식은 다음 구성 요소를 고려한 것입니다:

* `2hv`: 입력 및 출력 임베딩 (embedding + output projection)
* `2.5h²`: attention layer에서 발생하는 파라미터 (Q, K, V, O), 여기서는 K,V head의 개수가 Q의 1/4인 GQA 가정. QKVO 각각 $(h + 1/4h + 1/4h + h)$
* `3rh²`: feed-forward network layer 파라미터. Swiglu 사용 시 3개의 Linear network가 사용되기 때문에 3의 계수를 갖습니다.

**예시: LLaMA 3.1**

* **8B**: $h = 4{,}096$, $l = 32$, $v = 128{,}000$

  $$
  N \approx 2 \times 4{,}096 \times 128{,}000 \;+\; 32 \times (2.5 + 3 \times 3.467) \times 4{,}096^2 \approx 8\text{B}
  $$

* **70B**: $h = 8{,}192$, $l = 80$, $v = 128{,}000$

  $$
  N \approx 2 \times 8{,}192 \times 128{,}000 \;+\; 80 \times (2.5 + 3 \times 3.467) \times 8{,}192^2 \approx 70\text{B}
  $$


### **MoE (Mixture of Experts) 구조**
최근 주목받는 MoE 구조도 알아봅시다. MoE 구조는 FFN layer를 sparse 한 MoE 구조로 변경한 아키텍쳐이지만 attn layer에 들어가는 hidden size와 moe layer의 hidden size가 다르기 때문에 둘다 고려해 주어야합니다.

MoE 모델의 파라미터 수 계산:

* **E**: Expert 수
* **k**: 활성화되는 Expert 수 (top-k routing)
* **d\_model**: attention layer의 hidden size
* **d\_ff**: MoE FFN layer의 hidden size


$$
N_{MoE} = 2\,h\,v \;+\; l \times \bigl(2.5\,h^2 + E \times 3\,rh^2\bigr)
$$

**활성화 파라미터** (실제 연산에 사용되는 파라미터):
$$
N_{active} = 2\,h\,v \;+\; l \times \bigl(2.5\,h^2 + k \times 3\,rh^2\bigr)
$$

**예시: Qwen3**

---
## **FLOPS 계산법**
Dense 모델의 학습 시 전체 FLOPS는 다음과 같은 근사식을 통해 빠르게 계산할 수 있습니다:
* **F**: 전체 FLOPS (training 기준)
* $d$: 학습할 토큰 수 
* $t$: sequence length
* $h$: hidden dimension  
* $l$: 레이어 수
* $r$: FFN 확장 비율
* GEMM 연산이 대부분을 차지하며, backward pass는 forward의 2배이므로 전체에 3을 곱함

$$
F = 3 \times d \times l \times \bigl(6rh^2 + 2.5h^2 + 4th\bigr)
$$

위 식은 다음 구성 요소를 고려한 것입니다:
* `6rh²`: feed-forward network layer의 FLOPS. SwiGLU 가정 시 3개의 linear layer (W1: $2h \times rh$, W2: $2rh \times h$, W3: $2h \times rh$)
* `2.5h²`: attention layer의 linear projection FLOPS. GQA 가정 시 Q($h \times h$), K($h \times \tfrac{h}{4}$), V($h \times \tfrac{h}{4}$), O($h \times h$)
* `4th`: attention 연산 FLOPS. Q@K^T와 S@V 각각 $2th$의 연산량

**GEMM 연산에 대한 참고사항**: 행렬 곱셈 A(N×K) × B(K×M)은 N×K×M개의 곱셈-덧셈 쌍을 수행하므로 2×N×K×M FLOPS가 소요됩니다.

**예시: LLaMA 3.1, 1T 토큰 학습**
* **8B 모델** ($h = 4{,}096$, $l = 32$, $r = 3.467$, $t = 8{,}192$):
  $$
  F \approx 3 \times 10^{12} \times 32 \times (6 \times 3.467 \times 4{,}096^2 + 2.5 \times 4{,}096^2 + 4 \times 8{,}192 \times 4{,}096)
  $$
  $$
  \approx 5.04 \times 10^{22} \text{ FLOPS}
  $$



**참고**: 
* FFN 확장 비율 $r = \tfrac{8}{3}$, GQA 대신 MHA를 적용할 경우 ($2.5 \rightarrow 4$), DeepSeek에서 제안한 $ l \times d \times (72 * h^2 + 12 * t \times h)$ 공식과 정확히 일치합니다.
* 일반적인 Transformer 구조에서는 PaLM 논문의 "6N" 근사식(N은 파라미터 수)도 충분히 실용적입니다.

### **메모리 계산법 (학습)**
### **메모리 계산법 (추론)**
