
# 7강 복습 퀴즈

# 1번, 함수 파라미터
def calculate(a, b=5):
    if a > b:
        return a - b
    else:
        return b - a
    
result = calculate(3)
print(result * 10) # 20

print('=======================')

# 2번, 가변 인수, 키워드 가변 인수, 튜플, 딕셔너리
def process_tags(*tags, **attrs):
    tag_str = ", ".join(tags) # '파이썬, 정보처리기사, 합격'

    if 'prefix' in attrs:
        return f"[{attrs['prefix']}] {tag_str}"
    else:
        return tag_str

result = process_tags("파이썬", "정보처리기사", "합격", prefix="필수")
print(result)

print('=======================')

# 3번, lambda, filter, map, 딕셔너리
# 함수가 동시에 호출되면 안쪽(뒤쪽)부터 호출
menu = [
    {'item': '아메리카노', 'price': 1200},
    {'item': '딸기라떼', 'price': 2500},
    {'item': '초코케이크', 'price': 4000},
]

high_price_items = list(
            map(lambda r: r['item'], # '딸기라떼', '초코케이크'
            filter(lambda r: r['price'] >= 2000, menu)) # 아메리카노 걸러짐
    )

print(high_price_items)