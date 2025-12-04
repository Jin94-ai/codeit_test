# CodeIt 팀 협업 연습 프로젝트

GitHub를 이용한 5인 팀 협업을 연습하기 위한 프로젝트입니다.

## 프로젝트 소개

**목표**: GitHub 협업 워크플로우 마스터하기
- Branch 전략 이해 및 활용
- Pull Request 생성 및 Code Review
- Merge Conflict 해결
- Issue 기반 작업 흐름

**결과물**: 팀원 소개 웹페이지

## 프로젝트 구조

```
codeit_test/
├── index.html              # 메인 페이지
├── style.css               # 공통 스타일
├── README.md               # 이 파일
├── members/                # 각 팀원 프로필 (Jupyter Notebook)
│   ├── lee-jinseok/
│   │   └── profile.ipynb
│   ├── kim-minwoo/
│   │   └── profile.ipynb
│   ├── kim-boyoon/
│   │   └── profile.ipynb
│   ├── hwang-yumin/
│   │   └── profile.ipynb
│   └── kim-nayeon/
│       └── profile.ipynb
└── docs/
    ├── GITHUB_GUIDE.md         # GitHub 협업 가이드
    └── WORKFLOW_EXAMPLE.md     # 실전 예시
```

## 왜 Jupyter Notebook (.ipynb)?

- Python 코드와 마크다운을 함께 사용 가능
- GitHub에서 자동으로 렌더링
- 데이터 시각화 및 코드 예제 포함 가능
- 실제 프로젝트처럼 코드 실행 결과 확인 가능

## 학습 목표

### 1단계: 기본 워크플로우 (필수)
- [x] Repository Clone
- [ ] Branch 생성
- [ ] 파일 수정/추가
- [ ] Commit & Push
- [ ] Pull Request 생성
- [ ] Code Review
- [ ] Merge

### 2단계: 협업 실전 (중요)
- [ ] Feature Branch에서 작업
- [ ] 다른 사람 PR에 리뷰 남기기
- [ ] Merge Conflict 해결
- [ ] Rebase 활용

### 3단계: 고급 기능 (선택)
- [ ] Issue 생성 및 관리
- [ ] Project Board 활용
- [ ] GitHub Actions (CI/CD)

## 빠른 시작

### 1. Repository Clone

```bash
git clone https://github.com/Jin94-ai/codeit_test.git
cd codeit_test
```

### 2. 자신의 Feature Branch 생성

```bash
# 예시: member1이 작업하는 경우
git checkout -b feature/member1-profile
```

### 3. 파일 수정

- 자신의 폴더 생성: `members/your-name/`
- `members/your-name/profile.ipynb` 생성 (Jupyter Notebook 프로필)

### 4. Commit & Push

```bash
git add .
git commit -m "Add member1 profile card"
git push origin feature/member1-profile
```

### 5. Pull Request 생성

GitHub 웹사이트에서:
1. "Pull requests" 탭 클릭
2. "New pull request" 클릭
3. base: `main` ← compare: `feature/member1-profile`
4. 제목과 설명 작성
5. "Create pull request" 클릭

## 상세 가이드

- [GitHub 협업 가이드](docs/GITHUB_GUIDE.md) - 기본 개념 및 명령어
- [실전 워크플로우 예시](docs/WORKFLOW_EXAMPLE.md) - 단계별 실습

## 팀원

| 이름 | 역할 | 프로필 |
|------|------|--------|
| 이진석 | Backend Developer | [profile.ipynb](members/lee-jinseok/profile.ipynb) |
| 김민우 | Data Scientist | [profile.ipynb](members/kim-minwoo/profile.ipynb) |
| 김보윤 | Full Stack Developer | [profile.ipynb](members/kim-boyoon/profile.ipynb) |
| 황유민 | AI/ML Engineer | [profile.ipynb](members/hwang-yumin/profile.ipynb) |
| 김나연 | DevOps Engineer | [profile.ipynb](members/kim-nayeon/profile.ipynb) |

## Branch 전략

```
main (배포용 - 항상 안정적인 상태)
  ↑
develop (개발용 - 통합 브랜치)
  ↑
feature/member1-profile (개인 작업용)
feature/member2-profile
...
```

### Branch 네이밍 규칙

- `feature/작업내용`: 새 기능 추가
  - 예: `feature/member1-profile`
  - 예: `feature/update-styling`
- `fix/버그내용`: 버그 수정
  - 예: `fix/typo-in-readme`
- `docs/문서내용`: 문서 업데이트
  - 예: `docs/add-contribution-guide`

## Commit 메시지 규칙

```
type: 간결한 설명

상세 설명 (선택)

Co-Authored-By: Name <email>
```

**Type 종류:**
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅 (기능 변경 없음)
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정

**예시:**
```bash
git commit -m "feat: Add member1 profile card with skills section"
```

## 자주 발생하는 이슈

### Merge Conflict 해결

```bash
# 1. main 브랜치 최신 상태로 업데이트
git checkout main
git pull origin main

# 2. 내 feature 브랜치로 돌아가기
git checkout feature/member1-profile

# 3. main 브랜치 내용 가져오기 (conflict 발생!)
git merge main

# 4. conflict 파일 수정 후
git add .
git commit -m "Resolve merge conflict"
git push origin feature/member1-profile
```

### 실수로 main에 직접 commit한 경우

```bash
# 1. 새 브랜치 생성
git checkout -b feature/my-work

# 2. main을 이전 상태로 되돌리기
git checkout main
git reset --hard origin/main

# 3. 새 브랜치에서 작업 계속
git checkout feature/my-work
```

## 학습 리소스

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub Docs](https://docs.github.com)
- [생활코딩 - Git](https://opentutorials.org/course/3837)

## 도움이 필요하면?

1. [Issues](https://github.com/Jin94-ai/codeit_test/issues) 탭에서 질문 남기기
2. 팀원에게 멘션하기 (@username)
3. [GitHub Discussions](https://github.com/Jin94-ai/codeit_test/discussions) 활용

---

**Happy Coding!**
