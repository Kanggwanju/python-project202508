파이썬 개발환경 설정

Python 홈페이지에서 3.9 이상 버젼 다운로드

설치하면서 바로 Install Now 누르지 않기

아래 둘 다 체크
Use admin privileges when installing py.exe
Add python.exe to PATH

다운로드 끝나면 명령 프롬프트에서
python --version
pip --version

vdCode 확장 korean, python, Code Runner 설치

Code Runner 한국어 설정
설정 - code runner 검색 - 좌측 Run Code Configuration
- Code-runner: Executor Map - Setting.json에서 편집 클릭
- "python": "set PYTHONIOENCODING=utf-8 && python -u"으로 수정
