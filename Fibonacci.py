import timeit
import numpy as np
import math


# 递归实现
def fib_recur(n):
    assert n >= 0, "n > 0"
    if n <= 1:
        return n
    return fib_recur(n - 1) + fib_recur(n - 2)


# 递推实现
def fib_loop_for(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_loop_while(n):
    a, b = 0, 1
    while n > 0:
        a, b = b, a + b
        n -= 1
    return a


# 通项公式实现
def fib(n):
    if n >= 0:
        f = (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n) / math.sqrt(5)
        return int(f)
    else:
        pass


# print(fib_loop_while(0))


def fib_matrix(n):
    for i in range(n):
        res = pow((np.matrix([[1, 1], [1, 0]], dtype='int64')), i) * np.matrix([[1], [0]])
    print(int(res[0][0]))


# 使用矩阵计算斐波那契数列
def Fibonacci_Matrix_tool(n):
    Matrix = np.matrix("1 1;1 0", dtype='int64')
    # 返回是matrix类型
    return np.linalg.matrix_power(Matrix, n)


def Fibonacci_Matrix(n):
    result_list = []
    for i in range(0, n):
        result_list.append(np.array(Fibonacci_Matrix_tool(i))[0][0])
    return result_list


from timeit import Timer


def test(n):
    str_time_fib_loop_for = "fib_loop_for(" + str(n) + ")"
    T_fib_loop_for = Timer(str_time_fib_loop_for, "from __main__ import fib_loop_for")
    print(str_time_fib_loop_for, end='\n')
    print("Ans = " + str(fib_loop_for(n)), end='\n')
    print(T_fib_loop_for.timeit(number=10))

    str_time_fib_recur = "fib_recur(" + str(n) + ")"
    T_fib_recur = Timer(str_time_fib_recur, "from __main__ import fib_recur")
    print(str_time_fib_recur, end='\n')
    print("Ans = " + str(fib_recur(n)), end='\n')
    print(T_fib_recur.timeit(number=10))


# test(5)
# test(15)
# test(25)
# test(35)
# str_time_Fibonacci_Matrix = "Fibonacci_Matrix(" + str(n) + ")"
# T_Fibonacci_Matrix = Timer(str_time_Fibonacci_Matrix, "from __main__ import Fibonacci_Matrix")
# print(str_time_Fibonacci_Matrix, end='\n')
# print(T_Fibonacci_Matrix.timeit(), 3)

print(fib(5))
print(fib(15))
print(fib(25))
print(fib(35))