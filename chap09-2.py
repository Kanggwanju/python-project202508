
# 9강 복습 퀴즈

# 1번, 클래스, 생성자, 메서드

class Album:
    def __init__(self, title, artist):
        self.title = title
        self.artist = artist
        self.like = 0
    
    def add_like(self):
        self.like += 100

newjeans_album = Album("Get Up", "NewJeans")
newjeans_album.add_like()
newjeans_album.add_like()

print(newjeans_album.like)
print('=========================')


# 2번, 클래스, 생성자, 객체 배열, in

class Drink:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def get_info(self):
        return self.name + " : " + str(self.price) + "원"

drinks = [
    Drink("콜라", 1500),
    Drink("사이다", 1400),
    Drink("제로콜라", 1600)
]

total_price = 0
for d in drinks:
    if "콜라" in d.name: # in은 포함되어있다면 true ex) "콜라", "제로콜라"
        total_price += d.price

print(total_price)
print('=========================')


# 3번, 클래스, 생성자, 매서드

class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
        self.grade = ''
    
    def set_grade(self):
        if self.score >= 90:
            self.grade = 'A'
        elif self.score >= 80:
            self.grade = 'B'
        else:
            self.grade = 'C'

s1 = Student("카리나", 85)
s2 = Student("윈터", 95)
s1.set_grade()
s2.set_grade()

print(s1.grade)
print('=========================')


# 4번, 클래스, 생성자, 객체 배열, lambda, map, filter, join

class Song:
    def __init__(self, title, artist, duration):
        self.title = title
        self.artist = artist
        self.duration = duration

playlist = [
    Song("Ditto", "NewJeans", 185),
    Song("Spicy", "aespa", 200),
    Song("I AM", "IVE", 195)
]

long_play_titles = list(
    # 안쪽 필터링 -> map
    map(lambda s: s.title, filter(lambda s: s.duration >= 190, playlist))
)

# join: 리스트를 문자열로 바꿈
print(", ".join(long_play_titles))

