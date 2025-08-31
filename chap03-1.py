
a = 5

if a <= 10:
    print('hello')
    print('이 논리는 참입니다.')
print('메롱메롱')

print('=========')

my_favorite_teenieping = '믿어핑'

if my_favorite_teenieping == '하츄핑':
    print('역시 로열 티니핑! 프린세스의 베스트 프렌드!')
    print('오늘밤 주인공은 너야 너!!')
print('캐치 ! 티니핑! 사랑해요!') # 100% 실행코드

print('=========')

is_fearnot = False

if is_fearnot:
    print('FEARNOT 이시군요!')
else:
    print('죄송합니다 ㅠㅠ 피어나 멤버 전용 이벤트입니다.')    


print('=========')

fan_age = 13

if fan_age >= 19:
    print('성인 MY시군요! 에스파 음악방송에 바로 입장가능합니다.')
elif fan_age >= 15:
    print('청소년 MY시군요. 보호자 동의서가 필요합니다.')
elif fan_age >= 12:
    print('애매하긴 한데 보호자 동의서 내주시면 입장시켜드릴게요.')
else:
    print('Next Level은 아직... 15세 미만은 다음에 함께해요!!')

print('=========')

has_album = True
is_winner = True

if has_album:
    print('1차 조건 통과: 앨범을 구입하셨군요!')
    if is_winner:
        print('2차 조건 통과: 축하합니다. 팬 사인회에 당첨되었습니다.')
    else:
        print('아쉽지만.. 추첨에서 당첨되지 않았습니다.')
else:
    print('팬 사인회에 응모하시려면 앨범부터 구매해야 합니다.')

print('=========')

if has_album and is_winner:
    print('당첨!')
else:
    print('낙첨!')

# 단축 평가
# and 연산 -> A가 False면 B는 쳐다보지도 않고 전체 False
# or 연산 -> A가 True면 B는 쳐다보지도 않고 전체 True
score = 85
if score > 90 and score / 0 == 0:
    print('이 메세지는 볼 수 없어요.')
