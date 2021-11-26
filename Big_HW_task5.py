#!/usr/bin/env python
# coding: utf-8

# Петр Васильевич, директор ОАО "Рога и рога", собирается раздать премию всем менеджерам компании, он добрый и честный человек, поэтому хочет соблюсти следующие условия:
# - премия должна быть равной для всех менеджеров
# - должна быть максимально возможной и целой
# - должна быть выдана одной транзакцией с одного счета для каждого менеджера, без использования нескольких счетов для отправки одной премии
# 
# У Петра Васильевича открыто N корпоративных счетов, на которых лежат разные суммы денег Cn, а в компании работает M менеджеров.
# Необходимо выяснить максимальный размер премии, которую можно отправить с учетом условий. Если денег на счетах компании не хватит на то, чтобы выдать премию хотя бы по 1 у.е. - значит премии не будет, и нужно вывести 0. 
# 
# Входные данные (поступают в стандартный поток ввода)
# Первая строка - целые числа N и M через пробел (1≤N≤100 000, 1≤M≤100 000)
# Далее N строк, на каждой из которых одно целое число Cn (0≤Cn≤100 000 000)
# Проверка входных данных и обработка неправильных данных на входе не нужна, тестовые данные для проверки гарантированно подходят под описание выше
# 
# Выходные данные (ожидаются в стандартном потоке вывода)
# Одно целое число, максимально возможная премия

# In[54]:


def max_award(accounts, managers):
    
    Cn = [int(input()) for i in range(accounts)] #money on each (N) account
    Cn_sum = sum(Cn) #the overall money
    
    for i in range(Cn_sum // managers, 0, -1): #decreasing i as decreasing in possible awards
        temp = map(lambda int_division: int_division // i, Cn) #map==hidden for//lambda==hidden function
        temp = sum(temp) #number of people to receive award [Cn]
        
        if temp >= managers:
            return i
    return 0


# In[55]:


#test 1
N, M = (int(i) for i in input().split())
print(max_award(N, M))


# In[56]:


#test 3
N, M = (int(i) for i in input().split())
print(max_award(N, M))


# In[57]:


#test 2
N, M = (int(i) for i in input().split())
print(max_award(N, M))


# In[ ]:



