import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

def odesys(y, t, m1, m2, a, b, l0, c, g):
    l = ((8*(a**2)*(1 - np.cos(y[0]))) + (l0*(l0 - 4*a*np.sin(y[0]))))**(0.5)
    dy = np.zeros(4)
    dy[0] = y[2]  #dphi
    dy[1] = y[3]  #dksi

    a11 = a*((4/3) * m1 + m2)
    a12 = m2*b*np.sin(y[1] - y[0])
    a21 = a * np.sin(y[1] - y[0])
    a22 = b

    b1 = c * ((l0/l) - 1) * (4*a*np.sin(y[0]) - 2*l0*np.cos(y[0])) - ((m1 + m2)*g*np.cos(y[0])) - (m2*b*((y[3])**2)*np.cos(y[1]-y[0]))

    b2 = (a * ((y[3])**2) * np.cos(y[1] - y[0])) - (g * np.sin(y[1]))

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21) #dphi(t)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21) #dksi(t)

    return dy

m1 = 5
m2 = 0.5
a = b = l0 = 1
c = 25000
g = 9.80665
    
# функция для поворота
def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

# вспомогательные данные для заполнения массивов, кол-во шагов  и тд
Steps = 2001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

# Задаю начальные данные системы
phi0 = 0
ksi0 = math.pi / 18
dphi0 = dksi0 = 0
y0 = [phi0, ksi0, dphi0, dksi0]

Y = odeint(odesys, y0, t, (m1, m2, a, b, l0, c, g))

phi = Y[:,0]
ksi = Y[:,1]
dphi = Y[:,2]
dksi = Y[:,3]
ddphi = [odesys(y,t,m1, m2, a, b, l0, c, g)[2] for y,t in zip(Y,t)] #вычисляется угловое ускорение ddphi с использованием функции odesys для каждого момента времени t
ddksi = [odesys(y,t,m1, m2, a, b, l0, c, g)[3] for y,t in zip(Y,t)] #вычисляется угловое ускорение ddksi с использованием функции odesys для каждого момента времени t

RA = m2 * (g * np.cos(ksi) + b * (dksi**2) + a*(ddphi*np.cos(ksi - phi) + (dphi**2)*np.sin(ksi - phi)))

# создаю окно для отрисовки графиков и отрисовываю их
fig_for_graphs = plt.figure(figsize=[13,7])

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, ksi, color='Red')
ax_for_graphs.set_title("ksi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, RA, color='Green')
ax_for_graphs.set_title("RA(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

a = 2.5
lenDE = 2*a    #длина стержня DE
l = a * 1.8    #длина стержня АВ
l0 = 1.55 * lenDE    #высота, на которой закреплена пружина
D = np.array([0, 0])

# прорисовка пружины
K = 50
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

# Координаты точки Е
Ex = lenDE * np.cos(phi)
Ey = lenDE * np.sin(phi)

# смещение точки B относительно точки А
Bx = l * np.sin(ksi)
By = l * np.cos(ksi)

# подсчитываю отрезки, необходимые для вычисления угла наклона пружины
Spr_x = lenDE - Ex
Spr_y = l0 - Ey
# это длина пружины в каждый момент времени
length_Spr = (Spr_x**2 + Spr_y**2)**(0.5)

# Отрисовка окна, его параметризация
fig = plt.figure(figsize=[20, 10])
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')#чтобы оси были равнымич
ax.set_ylim([-8, 12])
ax.set_xlim([-6, 12])

# ПРУЖИНА

# получаю координаты пружины после поворота
Spr_x_L_fi, Spr_y_fi = Rot2D(X_Spr, Y_Spr, -(math.pi/2 + abs(math.atan2( Spr_x[0], Spr_y[0]))))
# задаю пружину уже после повторота, причём сразу перемещаю её в конечную позицию и растягиваю на длину
WArrow, = ax.plot(Spr_x_L_fi + lenDE, (Spr_y_fi*length_Spr[0]) + l0)
#крепёж для пружины
ax.plot(2*a, l0, color='black', linewidth=5, marker='o')
ax.plot([2*a-0.5, 2*a+0.5, 2*a, 2*a-0.5], [l0+0.7, l0+0.7, l0, l0+0.7], color='black', linewidth=2, )

ax.plot([-0.5, 0.5, 0, -0.5], [-0.5, -0.5, 0, -0.5], color='black', linewidth=2)
ax.plot([-0.75, 0.75], [-0.5, -0.5], color='black', linewidth=3)
# рисую оси
ax.plot([0, 0], [0, 8.25], color='red', linewidth=3, linestyle='solid', alpha=0.5, marker = '^')
ax.plot([0, 8.25], [0, 0], color='red', linewidth=3, linestyle='solid', alpha=0.5, marker = '>')
# рисую отрезок DE
Drawed_DE = ax.plot(np.array([0, Ex[0]]), np.array([0, Ey[0]]), color='black', linewidth=5, marker='o', markersize=8)[0]
# рисую отрезок АВ
Drawed_AB = ax.plot(np.array([Ex[0]/2, Ex[0]/2 + Bx[0]]), np.array([Ey[0]/2, Ey[0]/2 - By[0]]), color='black', linewidth=5, marker='o', markersize=8)[0]

def anima(i):

    # отрисовываю отрезок DE
    Drawed_DE.set_data(np.array([0, Ex[i]]), np.array([0, Ey[i]]))
    # получаю координаты пружины после поворота
    Spr_x_L_fi, Spr_y_fi = Rot2D(X_Spr*length_Spr[i], Y_Spr, -(math.pi/2 + abs(math.atan2(Spr_x[i], Spr_y[i]))))
    # задаю пружину уже после повторота, причём сразу перемещаю её в конечную позицию и растягиваю на длину
    WArrow.set_data(Spr_x_L_fi + lenDE, (Spr_y_fi) + l0)


    Drawed_AB.set_data(np.array([Ex[i]/2, Ex[i]/2 + Bx[i]]), np.array([Ey[i]/2, Ey[i]/2 - By[i]]))
    return Drawed_DE

anim = FuncAnimation(fig, anima, frames=len(t), interval=100, repeat=False)

plt.show()
