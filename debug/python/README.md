<!-- # Python Debugging with VSCode

> VSCode를 기준으로 Python 디버깅의 실전 팁을 정리합니다. 초보 개발자를 대상으로합니다.

### 반드시x999 디버거를 익혀라  
파이썬이 개발이 빠른 이유는 쉬운 문법과 캐스팅, garbage collector의 존재도 있지만 디버깅에 최적화 된 인터프리터 언어기 때문입니다. 

`pdb`, `ipdb`, `VSCode Debugger` 등 어떤 도구든 좋습니다. **디버깅 도구를 쓸 줄 모르면, 파이썬을 할 줄 아는게 아닙니다.**  

이걸 모르면 언제까지 `print()`만 치거나 `Jupyter Notebook`에서 코딩한 내용을 복사하는 초보 개발자에 머물게 됩니다.

---

### 파이썬은 빠르게, 깊게 볼 수 있다

- **하나의 버그만 고치지 마세요**  
  파이썬은 중단점에서 대부분의 상태를 관찰할 수 있습니다.  
  멈춘 그 시점에서 여러 문제를 함께 들여다볼 수 있어요.  

- **작은 코드 스니펫은 없어도 됩니다**  
  파이썬은 런타임 상태에서 직접 실험할 수 있습니다.  
  따로 테스트 코드를 만들지말고, 디버그 콘솔에서 바로 실험해보세요.

---

## ⚙️ VSCode 디버깅 실전 팁

### 기본 설정 (`.vscode/launch.json`)

create a launch.json file을 누릅니다.

```json
{
  "name": "Python: Run main.py",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}", // 지금 클릭되어있는 파일
  "console": "integratedTerminal"
}
```


## 🔍 실전 디버깅 팁

- **조건부 중단점**: 특정 조건(`i == 42`)일 때만 멈추게 하기
- **Watch 창**: 관심 있는 표현식 추적
- **Debug Console**: 실행 중인 변수와 객체 직접 조작 가능

```python
# 예시
my_tensor.shape
loss.backward()
model.layer[0].weight.mean()
```

task.json 처럼 활용하기
매번 실행할때마다 python a.py --arg1 --arg2 --arg3 을 입력하지마세요.


---

## ⚡ Accelerator 디버깅 (예정)

```bash
# CUDA 연산 에러 위치 정확히 찾기
CUDA_LAUNCH_BLOCKING=1 python train.py
```


좋아요. 지금 스타일을 유지하면서 실전적인 디버깅 상황별 가이드를 더해 아래와 같이 정리해드릴게요:

---

## 🛠️ 디버깅 상황별 팁

### 🧭 어디서 버그가 났더라?

* **VSCode의 Stack 창**을 활용하세요.
  에러가 발생했을 때, 함수 호출 스택을 통해 어떤 흐름으로 실행되었는지 추적할 수 있습니다.
* 스택 프레임을 클릭하면 해당 시점의 지역 변수, 파라미터도 함께 확인할 수 있고 여기 기준으로 debug consol을 사용할 수있습니다다


pylance를 활용하자
ctrl하고 클릭해서 들어갑시다.

---

### 🧩 제 코드는 `python` 명령어로 실행하지 않는데요?

* `python -m my_package.run` 같이 모듈 형태로 실행하는 경우, `launch.json`에서 아래처럼 설정하세요:

```json
{
  "name": "Python: Run as module",
  "type": "python",
  "request": "launch",
  "module": "my_package.run",
  "console": "integratedTerminal"
}
```
### 🧩 제 코드는 `python` 명령어로 실행하지 않는데요?


### + 팁 No module found 오류에 대해서
---
pip install -e .
sys.append


### 😕 제가 작성한 코드에서 오류가 난 게 아니에요

* 에러가 PyTorch, TensorRT, 기타 외부 라이브러리 안쪽에서 발생했다면,
  VSCode는 기본적으로 사용자가 작성한 코드만 추적합니다.

* **`justMyCode: false`** 설정을 통해 외부 코드까지 추적할 수 있습니다:

```json
{
  "justMyCode": false
}
```

* 단, C++/Rust 등의 native 라이브러리 내부에서 난 오류는 Python 디버거로는 추적할 수 없습니다.
  이런 경우는 C++/Rust 디버깅 도구를 활용해야 합니다:
  → [C++ 디버깅 가이드](../cpp/debug.md) / [Rust 디버깅 가이드](../rust/debug.md)

---

### 🔄 처음으로 돌아가야겠다… 내가 뭘 바꿨더라?

* `git`은 단순히 github에 코드 올리는 용도가 아닙니다.
  **변경사항 추적**에 가장 강력한 도구입니다.

* VSCode의 Source Control 창에서:

  * 초록색은 새로 추가한 코드
  * 파란색은 수정된 코드
    해당 줄에서 **우클릭 → Revert Changes** 하면 바로 되돌릴 수 있습니다. 어떤걸 바꿨는지 비교도 가능합니다.

* **커밋이 부담된다면, staging만 해도 좋습니다.**
  작업 전마다 `+` 눌러 스테이징만 해두면, 어떤 걸 바꿨는지 추적하기 편해집니다. -->