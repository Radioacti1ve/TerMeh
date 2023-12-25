import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Функция для поворота
def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

# Вспомогательные данные
Steps = 500
t_fin = 20
t = np.linspace(0, t_fin, Steps)
phi = np.sin(t)
ksi = np.cos(t)

a = 3
lenDE = 2*a    # Длина стержня DE
l = a * 1.8    # Длина стержня АВ
l0 = 1.55 * lenDE    # Высота, на которой закреплена пружина
D = np.array([0, 0])

# Для прорисовки пружины
K = 100
Sh = 0.4
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0

for i in range(K-2):
    X_Spr[i+1] = b*((i+1) - 1/2)
    Y_Spr[i+1] = Sh*(-1)**i

# Координаты точки E
Ex = lenDE * np.cos(phi)
Ey = lenDE * np.sin(phi)

# Смещение точки B относительно точки A
Bx = l * np.sin(ksi)
By = l * np.cos(ksi)

# Подсчитываю отрезки, необходимые для вычисления угла наклона пружины
Spr_x = lenDE - Ex
Spr_y = l0 - Ey

#длина пружины в каждый момент времени
length_Spr = (Spr_x**2 + Spr_y**2)**(0.5)

# Отрисовка окна (с указанием параметров)
fig = plt.figure(figsize=[20, 10])
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')  # для равности осей
ax.set_ylim([-8, 12])
ax.set_xlim([-6, 12])

# ПРУЖИНА
# Получаю координаты пружины после поворота
Spr_x_L_fi, Spr_y_fi = Rot2D(X_Spr, Y_Spr, -(math.pi/2 + abs(math.atan2(Spr_x[0], Spr_y[0]))))

# Задаю пружину уже после поворота, перемещая её в конечную позицию и растягивая на длину
WArrow, = ax.plot(Spr_x_L_fi + lenDE, (Spr_y_fi*length_Spr[0]) + l0)

# Крепёж для пружины
ax.plot(2*a, l0, color='black', linewidth=5, marker='o')
ax.plot([2*a-0.5, 2*a+0.5, 2*a, 2*a-0.5], [l0+0.7, l0+0.7, l0, l0+0.7], color='black', linewidth=2, )

ax.plot([-0.5, 0.5, 0, -0.5], [-0.5, -0.5, 0, -0.5], color='black', linewidth=2)
ax.plot([-0.75, 0.75], [-0.5, -0.5], color='black', linewidth=3)

# Рисую оси
ax.plot([0, 0], [0, 8.25], color='red', linewidth=3, linestyle='solid', alpha=0.5, marker='^')
ax.plot([0, 8.25], [0, 0], color='red', linewidth=3, linestyle='solid', alpha=0.5, marker='>')

# Рисую DE
Drawed_DE = ax.plot(np.array([0, Ex[0]]), np.array([0, Ey[0]]), color='black', linewidth=5, marker='o', markersize=8)[0]

# Рисую АВ
Drawed_AB = ax.plot(np.array([Ex[0]/2, Ex[0]/2 + Bx[0]]), np.array([Ey[0]/2, Ey[0]/2 - By[0]]), color='black', linewidth=5, marker='o', markersize=8)[0]

def anima(i):
    # ---
    # Отрисовываю отрезок DE
    Drawed_DE.set_data(np.array([0, Ex[i]]), np.array([0, Ey[i]]))
    # Получаю координаты пружины после поворота
    Spr_x_L_fi, Spr_y_fi = Rot2D(X_Spr*length_Spr[i], Y_Spr, -(math.pi/2 + abs(math.atan2(Spr_x[i], Spr_y[i]))))
    # Задаю пружину уже после поворота, причём сразу перемещаю её в конечную позицию и растягиваю на длину
    WArrow.set_data(Spr_x_L_fi + lenDE, (Spr_y_fi) + l0)

    Drawed_AB.set_data(np.array([Ex[i]/2, Ex[i]/2 + Bx[i]]), np.array([Ey[i]/2, Ey[i]/2 - By[i]]))
    return Drawed_DE

anim = FuncAnimation(fig, anima, frames=len(t), interval=100, repeat=False)

plt.show()
