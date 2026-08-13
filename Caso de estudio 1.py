import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parámetros físicos del problema
# ---------------------------------------------------------
g = 9.8      # aceleración de la gravedad [m/s^2]
v = 50.0     # rapidez horizontal del objeto [m/s]
H = 200.0    # altura del objeto sobre el suelo [m]
a = 30.0     # magnitud de la aceleración del motor del cohete [m/s^2]
v0 = 40.0    # rapidez inicial del cohete (lanzamiento vertical) [m/s]

# ---------------------------------------------------------
# Posición del objeto en función del tiempo
# ---------------------------------------------------------
def objeto(t):
    """
    Devuelve la posición (x_obj, y_obj) del objeto en el instante t.
    El objeto se mueve horizontalmente con rapidez constante v,
    a una altura fija H, pasando por x=0 en t=0.
    """
    x_obj = v * t
    y_obj = H
    return x_obj, y_obj

# ---------------------------------------------------------
# Sistema de ecuaciones diferenciales: f(t, s)
# ---------------------------------------------------------
def f(t, s):
    """
    Lado derecho del sistema de EDOs de primer orden.

    Estado s = [x_r, y_r, vx_r, vy_r]  (posición y velocidad del cohete)

    Devuelve ds/dt = [vx_r, vy_r, ax_r, ay_r]
    """
    x_r, y_r, vx_r, vy_r = s

    x_obj, y_obj = objeto(t)

    # Vector del cohete al objeto, y su magnitud (distancia)
    dx = x_obj - x_r
    dy = y_obj - y_r
    D = np.sqrt(dx**2 + dy**2)

    # Evitar división por cero si D es extremadamente pequeño
    if D < 1e-8:
        D = 1e-8

    # Componentes de la aceleración del motor (magnitud a, dirigida al objeto)
    ax_r = a * dx / D
    ay_r = a * dy / D - g   # se le resta la gravedad

    return np.array([vx_r, vy_r, ax_r, ay_r])

def rk4_paso(t, s, h):
    """
    Avanza la solución un paso de tamaño h usando Runge-Kutta de 4to orden.

    Parámetros
    ----------
    t : float
        Tiempo actual.
    s : array de 4 componentes
        Estado actual [x_r, y_r, vx_r, vy_r].
    h : float
        Tamaño del paso temporal.

    Devuelve
    --------
    t_nuevo : float
        Tiempo t + h.
    s_nuevo : array de 4 componentes
        Estado avanzado un paso.
    """
    k1 = f(t,         s)
    k2 = f(t + h/2,   s + (h/2) * k1)
    k3 = f(t + h/2,   s + (h/2) * k2)
    k4 = f(t + h,     s + h * k3)

    s_nuevo = s + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    t_nuevo = t + h

    return t_nuevo, s_nuevo

s0 = np.array([0.0, 0.0, 0.0, v0])   # x_r=0, y_r=0, vx_r=0, vy_r=v0
h = 0.01
t1, s1 = rk4_paso(0.0, s0, h)
print(t1, s1)