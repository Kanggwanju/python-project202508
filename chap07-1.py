# 함수 만들기 기본 공식
# def 함수이름(매개변수1, ...):
#     # 실행할 코드 (들여쓰기 필수!)
#     # ...
#     return 결과물 # 선택사항


# 함수 정의하기 (레시피 만들기)
def print_hello_newjeans():
    print('안녕하세요! 뉴진스입니다.')
    print('Attention! Who is it!?')

# 함수는 반드시 호출이라는 명령으로 코드를 실행시킨다.
print_hello_newjeans()

print('================')

def add(n1, n2):
    print(f'n1 + n2 = {n1 + n2}')
    return n1 + n2

result = add(10, 20)
print(result ** 2)

print('================')

def check_mbti(mbti):
    if (len(mbti) != 4):
        print('경고! mbti는 4글자여야 합니다.')
        return # 함수가 그대로 종료

    print(f'입력하신 mbti는 {mbti}')

check_mbti('ISFP')
check_mbti('ENTPX')

print('================')

import random

def generate_praise(idol_name):
    praise = [
        f"{idol_name} 미모 실화? 박물관에 가자",
        f"오늘부터 내 통장은 {idol_name}꺼!!",
        f"세상이 {idol_name}을 싫어하는건 다 억까다~"
    ]
    return random.choice(praise)

print(generate_praise('안유진'))
print(generate_praise('허윤진'))
