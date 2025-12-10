# scripts/

실행 스크립트 모음


## 주요 파일

| 파일 | 설명 | 사용법 | 담당 |
|:-----|:-----|:-------|:-----|
| `exc.sh` | 실행 파이프라인 생성, 환경설정 | 프로젝트 루트에서  `./scripts/exc.sh` 이후 `exc_pip` | Model Architect |
| `run.sh` | 실제 파이프라인 실행 | 자동 생성, 실행 | Model Architect |
| `Makefile` | 자동생성된 데이터, 결과 파일 삭제 | `make -f scripts/Makefile clean_data` | Model Architect |

## 설명

캐글API를 활용하여 자동으로 데이터를 다운받고 전처리하며 모델을 실행할 수 있도록하는 파이프라인 명령어 파일입니다. 모든 실행, 명령어는 프로젝트 루트에서 실행되는 것을 전제하여 작성되었습니다.

permission 관련 : chmod +x ./scripts/exc.sh
만약 `exc_pip`명령어가 없다고 오류가 나타난다면 `source ~/.bashrc` 코드를 실행해보고 안되면 알려주세요.
