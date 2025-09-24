
# 게임 캐릭터를 만들기 위한 설계도, 클래스
class Character:
    # pass # 나중에 작성할게, placehold

    # 메서드: 첫번째 매개변수로 self를 무조건 설정!
    def __init__(self, name, hp): # 생성자, 자동호출
        # 초기 데이터 설정
        self.name = name
        self.hp = hp
    
    # 자기 스펙을 보여주는 메서드
    def showInfo(self):
        print(f'캐릭터명: {self.name}, HP: {self.hp}')

    # 상대방을 공격하는 기능, 객체 상호작용
    def attack(self, target):
        print(f'{self.name}님이 {target.name}님을 때렸어요')
        target.hp -= 12;

# 설계도 자체론 아무일도 일어나지 않음
# print(Character)

# 객체 생성, self는 안 넣어줘도 된다.
warrior = Character('전사킹', 200)
wizard = Character('마법싸', 150)

# print(warrior)
# print(wizard)

# 메서드 호출
warrior.showInfo()
wizard.showInfo()

# 속성에 접근
print(warrior.name)
print(warrior.hp)

print('=================')

warrior.attack(wizard)
warrior.attack(wizard)
warrior.attack(wizard)

wizard.attack(warrior)

warrior.showInfo()
wizard.showInfo()

print('=================')

# 객체를 리스트로 관리
party = [
    warrior,
    wizard,
    Character('주차왕파킹', 170),
    Character('스티브호킹', 250)
]

for charac in party:
    print(f'이름: {charac.name}, 체력: {charac.hp}')

print('=================')

# hp가 180이 넘는 캐릭터만 골라내기
large_hp_characters = list(filter(lambda charac: charac.hp >= 180, party))
for charac in large_hp_characters:
    charac.showInfo()

print('=================')

# 모든 캐릭터의 이름만 추출하기
names = list(map(lambda charac: charac.name, party))
print(names)
