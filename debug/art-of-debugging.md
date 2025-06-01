# Art of Debugging

이 문서는 다음 내용을 소개하기 위해 작성되었습니다:

* 각 언어별로 버그를 빠르게 해결하는 방법 기술
* AI engineer가 자주 활용하는 코드 스니펫과 디버거 설정
* 복잡하거나 처음 보는 문제도 디버깅 가능하게 만드는 방법



## 핵심 원칙:  
**빠르게 반복하고, 작게 실험하라.**

1. **빠른 반복 (Quick debug cycles)**  
   프로그램을 실행하고 문제 지점에 도달하는 데 10분씩 걸린다면, 그 디버깅은 이미 실패다.
   이상적으로는 몇 초 안에 다시 테스트할 수 있어야 한다.  
   대기 시간은 혼란을 유발하고, 잘못된 가설을 테스트하게 만든다.

2. **작은 실험**
   single CPU, GPU 부터 시작해서 Multi Node 까지 차례로 넓혀나간다.


## 📚 참고

- [The-Art-Of-Debugging](https://github.com/stas00/the-art-of-debugging) by Stas Bekman
