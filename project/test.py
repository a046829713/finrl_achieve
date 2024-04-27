class Counter:
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self  # 实例本身就是迭代器

    def __next__(self):
        if self.current > self.high:
            raise StopIteration  # 停止迭代
        else:
            self.current += 1
            return self.current - 1

# 使用Counter
for number in Counter(1, 3):
    print(number)
