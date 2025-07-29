
로깅 활성화

torch._logging.set_logs(distributed=logging.DEBUG)
이나 환경변수로 설정할 수 있다.

하지만 안되는 것도 존재하는데 c10 로 시작하는 로거들이다.
import 할 때 Null-handler를 붙이는데 로거 이름에도 붙여버려 사실상 Null-handler로 고정이되어있다. 쓰임새를 잘 모르겠다. meta 내부적으로 사용하는 것으로 추측만 할 뿐이다. (https://github.com/pytorch/pytorch/pull/121352)

따라서 수동으로 바꿔준다.


pytorch/torch/distributed/checkpoint/logging_handlers.py
```
import logging

from torch.distributed.logging_handlers import _log_handlers


__all__: list[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.StreamHandler(), // DCP_LOGGER_NAME: logging.NullHandler()에서 변경
    }
)
```

/home/julio/HY/pytorch/torch/distributed/checkpoint/utils.py
```

# TODO: integrate with distributed logging flag
ENABLE_PROFILE = True

```

save_plan??
global_plan??

기능을 확인해보자

디버깅으로 찾아간다.

pylance

