import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cálculo de fatorial usando recursividade
def fatorial(n):
    if (n == 0):
        return (1)
    return (n*fatorial(n-1))

#Calcula o cosseno utilizando a série de McLaurin
def cos(x):
    termo = 1
    i = 0
    resultado = 0

    #Calcula o módulo do erro e usa como condição de parada o erro de 2^-30 (Calcula até a soma do termo ser menor que 2^-30)
    while (abs(termo) > 2**-30):
            termo = (pow(-1,i)*(pow(x,2*i)))/fatorial(2*i)
            resultado += termo
            i += 1

    return (resultado)

print(cos(1))

def runge_kutta(h, xa):
    n = int(xa / h)
    x_values = np.zeros(n+1)
    y_values = np.zeros(n+1)
    v_values = np.zeros(n+1)

    # Condições iniciais
    x_values[0] = 0
    y_values[0] = -1
    v_values[0] = 0

    # Método de Runge-Kutta de quarta ordem
    for i in range(n):
        k1y = h * v_values[i]
        k1v = h * (-y_values[i] / cos(y_values[i]))

        k2y = h * (v_values[i] + 0.5 * k1v)
        k2v = h * (-(y_values[i] + 0.5 * k1y) / cos(y_values[i] + 0.5 * k1y))

        k3y = h * (v_values[i] + 0.5 * k2v)
        k3v = h * (-(y_values[i] + 0.5 * k2y) / cos(y_values[i] + 0.5 * k2y))

        k4y = h * (v_values[i] + k3v)
        k4v = h * (-(y_values[i] + k3y) / cos(y_values[i] + k3y))

        x_values[i+1] = x_values[i] + h
        y_values[i+1] = y_values[i] + (k1y + 2*k2y + 2*k3y + k4y) / 6
        v_values[i+1] = v_values[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6

    return x_values, y_values, v_values

def preditor_corretor(h, xa):
    n = int(xa / h)
    x_values = np.zeros(n+1)
    y_values = np.zeros(n+1)
    v_values = np.zeros(n+1)

    # Condições iniciais
    x_values[0] = 0
    y_values[0] = -1
    v_values[0] = 0

    # Método de Runge-Kutta de quarta ordem para inicializar e obter a estimativa
    x_values, y_values, v_values = runge_kutta(h, xa)

    # Corrector
    for i in range(4, n):
        x_values[i+1] = x_values[i] + h

        # Corretor usando Adams-Moulton
        y_values[i+1] = y_values[i] + (h/24)*(9*v_values[i+1] + 19*v_values[i] - 5*v_values[i-1] + v_values[i-2])

    return x_values, y_values, v_values

# Parâmetros
h = 0.01  # Tamanho do passo
xa = 5.01  # Ponto final

# Resolvendo o sistema de EDOs usando Runge-Kutta
x_values_rk, y_values_rk, v_values = runge_kutta(h, xa)
x_values_am, y_values_am, v_values = preditor_corretor(h, xa)

# Resolvendo o sistema de EDOs usando Adams-Moulton
#x_values_am, y_values_am = adams_moulton(h, xa)

# Plotando a solução
plt.plot(x_values_am, y_values_am, label='Adams-Moulton')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()

adams = {'x': x_values_am, 'y': y_values_am}
dfadams = pd.DataFrame(data=adams)
dfadams

def resolve_sistema(A, b):
    n = len(b)
    x = np.zeros(n)

    # Eliminação direta
    for i in range(1, n):
        m = A[i][i-1] / A[i-1][i-1]
        A[i] -= m * A[i-1]
        b[i] -= m * b[i-1]

    # Substituição reversa
    x[n-1] = b[n-1] / A[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]

    return x

def spline_cubica(x_data, y_data):
    n = len(x_data)
    h = np.diff(x_data)
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(1, n-1):
        A[i, i-1:i+2] = [h[i-1], 2*(h[i-1] + h[i]), h[i]]
        b[i] = 3 * ((y_data[i+1] - y_data[i]) / h[i] - (y_data[i] - y_data[i-1]) / h[i-1])

    # Condições de contorno naturais
    A[0, :2] = [2, 1]
    A[n-1, n-2:] = [1, 2]

    # Resolver o sistema tridiagonal para obter os coeficientes c
    c = resolve_sistema(A, b)

    # Calcular os coeficientes a, b, d
    a = y_data[:-1]
    b = (y_data[1:] - y_data[:-1]) / h - h * (c[:-1] + 2 * c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    # Retorna a função interpolante
    def interpola(x):
        result = np.zeros_like(x)

        for i in range(n-1):
          #Matriz booleana que indica se cada ponto em x está dentro do intervalo [x_data[i], x_data[i+1]].
            mask = np.logical_and(x_data[i] <= x, x <= x_data[i+1])
            result[mask] = a[i] + b[i] * (x[mask] - x_data[i]) + c[i] * (x[mask] - x_data[i])**2 + d[i] * (x[mask] - x_data[i])**3

        return result

    return interpola

x_data, y_data, v_data = preditor_corretor(h, xa)

# Cria a função interpolante spline cúbica
spline_interpolator = spline_cubica(x_data, y_data)

# Avalia a spline cúbica em pontos intermediários
x_interp = np.linspace(0, 4.99, 100)
y_interp = spline_interpolator(x_interp)

# Cria a função interpolante spline cúbica
spline_interpolator = spline_cubica(x_data, v_data)

# Avalia a spline cúbica em pontos intermediários
x_interp = np.linspace(0, 4.99, 100)
y_der = spline_interpolator(x_interp)



# Plotar resultados
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'o', label='Pontos dados')
plt.plot(x_interp, y_interp, label='Spline Interpolante')
plt.plot(x_interp, y_der, label='Spline Interpolante Derivada')
plt.title('Cubic Spline Interpolante')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        dfdx = (f(x + tol) - f(x - tol)) / (2 * tol)  # Aproximação numérica da derivada
        x_new = x - f(x) / dfdx
        if np.abs(x_new - x) < tol: #Se < Que tolerância retorna pra poder fazer o append nas raízes
            return x_new, dfdx
        x = x_new
    raise ValueError("O método de Newton-Raphson não convergiu.")

# Função que representa a spline cúbica interpolante
spline_interpolator = spline_cubica(x_data, y_data)

# Função para encontrar as raízes (pontos onde y = 0) e derivadas
def raizes_derivadas(x_range):
    roots = []
    derivatives = []
    for x_start, x_end in zip(x_range[:-1], x_range[1:]):
        try:
            # Encontrar uma raiz dentro de cada intervalo
            x_root, dfdx = newton_raphson(spline_interpolator, (x_start + x_end) / 2) #Utiliza newton-raphson no spline
            roots.append(x_root) #Append das raízes
            derivatives.append(dfdx)
        except ValueError:
            pass  # Ignorar se o método de Newton-Raphson não convergir no intervalo
    return np.array(roots), np.array(derivatives)

# Intervalo onde procura as raízes
x_range = np.linspace(0, 6, 1000)

# Encontrar as raízes e derivadas
roots, derivatives = raizes_derivadas(x_range)

# Avalia y nos pontos encontrados
y_roots = spline_interpolator(roots)

df = pd.DataFrame({
    'Raiz': roots,
    'Valor da Função': y_roots,
    'Derivada': derivatives
})

df

def float_binario(value):
    # Converte um número de ponto flutuante para representação binária
    if value == 0:
        return '0 00000 00000000000000000000'

    # Lida com o sinal
    sign_bit = '1' if value < 0 else '0'
    value = abs(value)

    # Converte a parte inteira
    int_part = format(int(value), 'b')

    # Separa a parte inteira da parte fracionária
    int_part, frac_part = int_part[:-23], int_part[-23:]

    # Calcula o expoente
    exponent = len(int_part) + 15

    # Converte o expoente para binário
    exponent_bits = format(exponent, '05b')

    # Converte a parte fracionária com 5 casas decimais
    frac_part += '0' * (23 - len(frac_part))
    frac_part = frac_part[:5]

    # Concatena os bits para formar a representação binária final
    binary_representation = f'{sign_bit} {exponent_bits} {frac_part}'

    return binary_representation

def df_binario(df):
    df_binary = pd.DataFrame()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df_binary[column + 'Binario'] = df[column].apply(float_binario)
        else:
            df_binary[column] = df[column]
    return df_binary

# Transformar o DataFrame para representação binária
dffinal = df_binario(df)

# Imprimir o DataFrame com representação binária
print(dffinal)

from scipy.interpolate import CubicSpline
x_values, y_values = preditor_corretor(h, xa)

# Interpolar os dados usando spline cúbico
cs = CubicSpline(x_values, y_values, bc_type='natural')

# Avaliar a função interpolante em pontos intermediários
x_interp = np.linspace(0, xa, 100)
y_interp = cs(x_interp)
y_interp_prime = cs(x_interp, 1)  # Primeira derivada

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'o', label='Pontos da solução')
plt.plot(x_interp, y_interp, label='Spline Cúbico Interpolante')
plt.plot(x_interp, y_interp_prime, label='Derivada do Spline Cúbico')
plt.title('Spline Cúbico Interpolante e Derivada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()