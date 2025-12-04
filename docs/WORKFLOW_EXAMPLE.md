# 실전 워크플로우 예시

실제 작업 시나리오를 따라하며 GitHub 협업 마스터하기

---

## 목차

1. [시나리오 1: 첫 번째 프로필 추가](#시나리오-1-첫-번째-프로필-추가)
2. [시나리오 2: 다른 사람과 동시 작업](#시나리오-2-다른-사람과-동시-작업)
3. [시나리오 3: Merge Conflict 해결](#시나리오-3-merge-conflict-해결)
4. [시나리오 4: 리뷰 피드백 반영](#시나리오-4-리뷰-피드백-반영)
5. [시나리오 5: Issue 기반 작업](#시나리오-5-issue-기반-작업)

---

## 시나리오 1: 첫 번째 프로필 추가

**상황**: Member 1이 자신의 프로필 카드를 처음 추가합니다.

### Step-by-Step

#### 1. Repository Clone (최초 1회만)

```bash
# 터미널 열기
cd Desktop  # 작업할 폴더로 이동

# Repository clone
git clone https://github.com/Jin94-ai/codeit_test.git

# 폴더 진입
cd codeit_test

# 잘 받아졌는지 확인
ls
# 출력: index.html  style.css  README.md  docs/  members/
```

#### 2. 새 Branch 생성

```bash
# 현재 브랜치 확인 (main에 있어야 함)
git branch
# 출력: * main

# 최신 상태로 업데이트 (습관화!)
git pull origin main

# 새 브랜치 생성 및 이동
git checkout -b feature/member1-profile

# 확인
git branch
# 출력:
#   main
# * feature/member1-profile
```

#### 3. 파일 수정

**a) 자신의 폴더 생성**

```bash
mkdir members/your-name
```

**b) Jupyter Notebook 프로필 생성**

`members/your-name/profile.ipynb` 파일을 생성합니다.

Jupyter Notebook 또는 VSCode에서:

```python
# 마크다운 셀 추가
```
```markdown
# 김코딩 - Python Developer

## About Me
안녕하세요! Python 개발자 김코딩입니다.

## Skills
- Python
- Django
- FastAPI
```

```python
# 코드 셀 추가 - 간단한 소개 코드
def introduce():
    profile = {
        'name': '김코딩',
        'role': 'Python Developer',
        'skills': ['Python', 'Django', 'FastAPI']
    }
    return profile

print(introduce())
```

**참고**: 실제 팀원 프로필 예시
- [이진석](https://github.com/Jin94-ai/codeit_test/blob/main/members/lee-jinseok/profile.ipynb)
- [김민우](https://github.com/Jin94-ai/codeit_test/blob/main/members/kim-minwoo/profile.ipynb)
- [김보윤](https://github.com/Jin94-ai/codeit_test/blob/main/members/kim-boyoon/profile.ipynb)
- [황유민](https://github.com/Jin94-ai/codeit_test/blob/main/members/hwang-yumin/profile.ipynb)
- [김나연](https://github.com/Jin94-ai/codeit_test/blob/main/members/kim-nayeon/profile.ipynb)

#### 4. 변경사항 확인

```bash
# 무엇이 바뀌었는지 확인
git status
# 출력:
# untracked:  members/your-name/
# untracked:  members/your-name/profile.ipynb

# Jupyter Notebook 파일은 JSON 형식이므로 diff가 복잡합니다
# GitHub에서 시각적으로 확인하는 것을 권장합니다
```

#### 5. Commit

```bash
# 파일 스테이징
git add members/your-name/profile.ipynb

# Commit (의미 있는 메시지!)
git commit -m "feat: Add my profile with Python examples

- Created Jupyter Notebook profile
- Added Python code examples
- Included data visualization"

# Commit 확인
git log --oneline -1
```

#### 6. Push

```bash
# 원격 저장소에 올리기
git push origin feature/member1-profile

# 출력:
# To https://github.com/Jin94-ai/codeit_test.git
#  * [new branch]      feature/member1-profile -> feature/member1-profile
```

#### 7. Pull Request 생성

**GitHub 웹사이트에서:**

1. Repository 페이지로 이동
2. 노란색 배너 "Compare & pull request" 클릭
   (또는 "Pull requests" 탭 → "New pull request")

3. PR 정보 입력:
   ```
   Title: Add member1 profile (김코딩)

   Description:
   ## 변경 사항
   - ✅ index.html에 프로필 카드 추가
   - ✅ members/member1.html 생성
   - ✅ 스킬 및 프로젝트 소개 포함

   ## 스크린샷
   (웹페이지 캡처 이미지 첨부)

   ## 체크리스트
   - [x] 로컬에서 브라우저 테스트 완료
   - [x] 반응형 디자인 확인
   - [x] 링크 동작 확인
   ```

4. Reviewers 지정 (팀원들)

5. "Create pull request" 클릭

#### 8. 팀원 리뷰 대기

팀원들이 코드를 리뷰하고 "Approve" 또는 코멘트를 남깁니다.

#### 9. Merge

Approve를 받으면:
1. "Merge pull request" 클릭
2. "Confirm merge" 클릭
3. "Delete branch" (선택사항)

축하합니다! 첫 번째 PR이 merge되었습니다!

#### 10. 정리

```bash
# main 브랜치로 이동
git checkout main

# 최신 상태로 업데이트 (자신의 변경사항 포함)
git pull origin main

# 사용한 브랜치 삭제 (선택)
git branch -d feature/member1-profile

# 확인
git log --oneline -3
```

---

## 시나리오 2: 다른 사람과 동시 작업

**상황**: Member 2가 Member 1과 동시에 작업하지만 충돌 없이 진행

### 핵심 포인트

- ✅ 각자 다른 브랜치에서 작업
- ✅ 각자 다른 파일 수정 (member1.html vs member2.html)
- ✅ index.html은 다른 위치에 카드 추가

### Member 2의 작업 흐름

```bash
# 1. 최신 main 가져오기
git checkout main
git pull origin main

# 2. 새 브랜치 생성
git checkout -b feature/member2-profile

# 3. 파일 수정
# - members/member2-name/ 폴더 생성
# - members/member2-name/profile.ipynb 생성

# 4. Commit & Push
git add members/member2-name/
git commit -m "feat: Add member2 profile with data analysis examples"
git push origin feature/member2-profile

# 5. PR 생성 (GitHub에서)
```

### 결과

- Member 1 PR: merge 완료 ✅
- Member 2 PR: 충돌 없이 merge 가능 ✅

**왜 충돌이 없을까?**
- 각자 다른 폴더에서 작업했기 때문!
  - Member 1: `members/member1-name/`
  - Member 2: `members/member2-name/`
- 독립적인 파일이므로 Git이 자동으로 병합 가능

---

## 시나리오 3: Merge Conflict 해결

**상황**: Member 3과 Member 4가 같은 줄을 수정해서 충돌 발생

### 충돌 발생 시나리오

**Member 3 작업:**
```html
<!-- index.html 50번째 줄 -->
<h2>팀원 소개</h2>
```
→ Commit & Push → PR 생성 → **Merge 완료 ✅**

**Member 4 작업:** (동시에 진행)
```html
<!-- index.html 50번째 줄 -->
<h2>우리 팀을 소개합니다</h2>
```
→ Commit & Push → PR 생성 → **Conflict 발생! ❌**

### 해결 과정 (Member 4)

#### 1. 충돌 확인

GitHub PR 페이지에서:
```
❌ This branch has conflicts that must be resolved
```

#### 2. 로컬에서 해결

```bash
# main의 최신 내용 가져오기
git checkout main
git pull origin main

# 내 브랜치로 돌아가기
git checkout feature/member4-profile

# main 내용 merge (충돌 발생!)
git merge main

# 출력:
# Auto-merging index.html
# CONFLICT (content): Merge conflict in index.html
# Automatic merge failed; fix conflicts and then commit the result.
```

#### 3. 충돌 파일 수정

`index.html` 열면:

```html
<<<<<<< HEAD (내 브랜치)
<h2>우리 팀을 소개합니다</h2>
=======
<h2>팀원 소개</h2>
>>>>>>> main
```

**수정 방법:**

**Option A: 하나 선택**
```html
<h2>팀원 소개</h2>
```

**Option B: 조합**
```html
<h2>우리 팀원을 소개합니다</h2>
```

**Option C: 팀에 물어보기**
```bash
# Issue 생성 또는 팀 채팅
"어떤 제목이 좋을까요?"
```

#### 4. 해결 완료

```bash
# 수정 완료 후
git add index.html

# Merge commit 생성
git commit -m "Resolve merge conflict in index.html

- Chose '팀원 소개' as the final title
- Discussed with team on Issue #5"

# Push
git push origin feature/member4-profile
```

#### 5. PR 확인

GitHub PR 페이지 새로고침:
```
✅ This branch has no conflicts with the base branch
✅ Merging can be performed automatically
```

→ 이제 Merge 가능!

---

## 시나리오 4: 리뷰 피드백 반영

**상황**: Member 5의 PR에 리뷰 코멘트가 달림

### 리뷰 코멘트 예시

```
Reviewer: @member2
Line 23 in index.html:

"class 이름을 'member-card'로 통일하면
일관성이 있을 것 같아요!"

Suggestion:
- <div class="profile-box">
+ <div class="member-card">
```

### Member 5의 대응

#### Option A: 수정 반영

```bash
# 1. 같은 브랜치에서 수정
# (feature/member5-profile 브랜치에 그대로 있음)

# 2. 파일 수정
# index.html의 class 이름 변경

# 3. Commit
git add index.html
git commit -m "Apply review feedback: Standardize class name to 'member-card'"

# 4. Push
git push origin feature/member5-profile
```

→ PR에 자동으로 새 commit 추가됨!

#### Option B: 토론

PR 코멘트에 답글:
```
"좋은 의견 감사합니다!
그런데 'profile-box'가 더 명확하지 않을까요?
다른 분들 의견은 어떠신가요? @member1 @member3"
```

→ 팀 토론 후 결정

### 리뷰어의 최종 Approve

```
✅ Looks good to me! (LGTM)
Approved
```

---

## 시나리오 5: Issue 기반 작업

**상황**: 새로운 기능을 Issue로 제안하고 구현

### 1. Issue 생성

GitHub "Issues" 탭에서:

```
Title: 프로필 카드에 소셜 링크 추가

Labels: enhancement, good first issue

Description:
## 제안 내용
각 멤버의 GitHub, LinkedIn 링크를 프로필 카드에 추가

## 구현 아이디어
- FontAwesome 아이콘 사용
- 카드 하단에 배치
- 새 탭에서 열리도록 (target="_blank")

## 예상 파일
- index.html (카드 HTML 수정)
- style.css (아이콘 스타일 추가)
- members/*.html (상세 페이지 업데이트)

## 담당
자원자를 기다립니다!
```

→ Issue #10 생성됨

### 2. Issue 할당

Member 3이 댓글:
```
"제가 해볼게요! Assign 부탁드립니다."
```

→ Assignees에 @member3 추가

### 3. Branch 생성 (Issue 번호 포함)

```bash
git checkout main
git pull origin main
git checkout -b feature/add-social-links-#10
```

### 4. 구현

```html
<!-- index.html -->
<div class="member-card">
    <!-- 기존 내용 -->
    <div class="social-links">
        <a href="https://github.com/member1" target="_blank">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://linkedin.com/in/member1" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
    </div>
</div>
```

```css
/* style.css */
.social-links {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1rem;
}

.social-links a {
    color: #667eea;
    font-size: 1.5rem;
    transition: color 0.3s ease;
}

.social-links a:hover {
    color: #5568d3;
}
```

### 5. Commit & Push

```bash
git add index.html style.css
git commit -m "feat: Add social media links to profile cards

Implements GitHub and LinkedIn links with FontAwesome icons.
Links open in new tab for better UX.

Closes #10"

git push origin feature/add-social-links-#10
```

### 6. PR 생성 (Issue 자동 연결)

```
Title: Add social media links to profile cards

Description:
Closes #10

## 구현 내용
- ✅ GitHub, LinkedIn 아이콘 추가
- ✅ 새 탭에서 열리도록 설정
- ✅ 호버 애니메이션 추가

## 스크린샷
(이미지 첨부)
```

### 7. Merge 시 Issue 자동 Close

PR이 merge되면:
- Issue #10이 자동으로 닫힘
- "Closed by PR #15" 표시

---

## 학습 포인트 정리

| 시나리오 | 핵심 개념 | 실무 활용도 |
|----------|-----------|-------------|
| 1. 첫 프로필 추가 | 기본 워크플로우 | ⭐⭐⭐⭐⭐ |
| 2. 동시 작업 | Branch 분리 | ⭐⭐⭐⭐⭐ |
| 3. Conflict 해결 | Merge 충돌 | ⭐⭐⭐⭐ |
| 4. 피드백 반영 | Code Review | ⭐⭐⭐⭐⭐ |
| 5. Issue 기반 작업 | 체계적 협업 | ⭐⭐⭐⭐⭐ |

---

## 다음 단계

1. ✅ 이 예시들을 실제로 따라해보기
2. ✅ 팀원들과 함께 연습하기
3. ✅ 자신만의 작업 패턴 찾기

---

## 팁

- **작은 단위로 자주 Commit**: 한 번에 너무 많이 변경하지 말기
- **의미 있는 메시지**: 나중에 봐도 이해 가능하게
- **적극적인 소통**: 막히면 바로 물어보기
- **일찍 자주 Push**: 백업 효과 + 팀원에게 진행 상황 공유

**Happy Collaborating!**
