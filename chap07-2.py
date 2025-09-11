
# 매개변수 기본값
def cafe_order(order='아메리카노'):
    print(f'주문하신 {order} 나왔습니다!')

cafe_order('카푸치노')
cafe_order()

print('================')


def profile(name, group):
    print(f'안녕하세요, {group}의 {name}입니다!')

# 위치 인수, 순서를 맞춰야 함 (positional argument)
profile('에스파', '지젤')
profile('지젤', '에스파')

# 키워드 인수, 이름을 실수로 잘못 적으면 오류 발생
profile(group='에스파', name='카리나')

print('================')


# 가변 인수 *args (모든 위치 인수들을 튜플로 묶어서 받음)
def show_my_friends(*friends):
    print(f'내 친구들을 소개할게! 총 {len(friends)}명이야!')
    for friend in friends:
        print(friend)
    print('=========')

show_my_friends('윈터', '닝닝')
show_my_friends('안유진', '가을', '이서', '레이')

print('================')


## 키워드 가변 인수 **kwargs (모든 키워드 인수들을 딕셔너리로 묶어서 받음)
def custom_order(**options):
    print('---- 주문서 ----')
    # print(options)
    for key, value in options.items():
        print(f'{key}: {value}')

custom_order(main='피자', topping='치즈', side='감자튀김')
custom_order(idol='BTS', song='dynamite', a_word='사랑해요')


# 함수 매개변수에는 엄격한 순서 규칙이 존재 (어기면 에러)
# 일반위치 -> 기본값 -> *args -> **kwargs
# 암기법: 일.기.아.키

