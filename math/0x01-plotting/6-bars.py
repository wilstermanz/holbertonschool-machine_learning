#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
apples, bananas, oranges, peaches = fruit[0], fruit[1], fruit[2], fruit[3]

plt.bar(people, apples,
        color='red',
        label='apples',
        width=0.5
        )
plt.bar(people, bananas,
        bottom=apples,
        color='yellow',
        label='bananas',
        width=0.5
        )
plt.bar(people, oranges,
        bottom=apples + bananas,
        color='#ff8000',
        label='oranges',
        width=0.5
        )
plt.bar(people, peaches,
        bottom=apples + bananas + oranges,
        color='#ffe5b4',
        label='peaches',
        width=0.5
        )
plt.title('Number of Fruit per Person')
plt.yticks(range(0, 81, 10))
plt.ylabel('Quantity of fruit')
plt.legend()

plt.show()
