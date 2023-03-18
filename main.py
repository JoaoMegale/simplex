import numpy as np
import sys

# leitura do arquivo e atribuição das variaveis
with open(sys.argv[1], "r") as arq:
    num_rest, num_var = [int(x) for x in next(arq).split()]
    c = np.zeros(0)
    c = np.append(c, [int(x) for x in arq.readline().split()])
    A = np.zeros((num_rest, num_var + 1))
    i = 0
    for line in arq:
        A[i] = [int(x) for x in line.split()]
        i = i + 1

b = A[:, num_var]
b_negativos = np.where(b < 0)[0]
b = b[np.newaxis]
A = np.delete(A, num_var, 1)

# cria o tableau
def cria_tableau(A, b2, c2):
    folgas = np.identity(num_rest)
    folgas_c = np.zeros(num_rest + 1)
    b = np.copy(b2)
    c = np.copy(c2)
    c = c * -1
    c = np.concatenate((c, folgas_c), axis=0)[np.newaxis]

    # junta tudo no tableau
    tableau = np.concatenate((A, folgas), axis=1)
    tableau = np.concatenate((tableau, b.T), axis=1)
    tableau = np.concatenate((c, tableau), axis=0)

    # adiciona VERO
    vero = np.identity(num_rest)
    vero_c = np.zeros(num_rest)[np.newaxis]
    vero = np.concatenate((vero_c, vero), axis=0)
    tableau = np.concatenate((vero, tableau), axis=1)

    # transforma os negativos de b em positivos
    for i in range(len(b_negativos)):
        tableau[b_negativos[i] + 1] *= -1

    return tableau

# cria pl auxiliar
def pl_auxiliar(A, b, c):
    c2 = np.zeros(np.shape(c)[0])
    tableau = cria_tableau(A, b, c2)
    b2 = tableau[:, np.shape(tableau)[1] - 1][np.newaxis]
    c_aux = np.ones(num_rest)
    c_aux = c_aux[np.newaxis]
    id = np.identity(num_rest)
    aux = np.concatenate((c_aux, id), axis=0)
    tableau = np.concatenate((tableau[:, 0 : np.shape(tableau)[1] - 1], aux), axis=1)
    tableau = np.concatenate((tableau, b2.T), axis=1)
    return tableau

# põe a pl auxiliar em base canonica para que possa fazer o simplex
def base_canonica_pl_auxiliar(pl):
    matriz = np.copy(pl)
    for i in range(num_rest):
        matriz[0] = matriz[0] - matriz[i + 1]
    return matriz

# encontra o valor do pivot e sua posição no tableau
def find_pivot(m):
    matriz = np.copy(m)
    num_cols = matriz.shape[1]
    vetor_b = matriz[0][num_rest:]
    negativos = np.where(vetor_b < 0)[0]
    pivot_encontrado = 0
    for j in range(len(negativos)):
        coluna_pivot = negativos[j] + num_rest
        linhas = []
        results = []
        for i in range(num_rest):
            if matriz[i + 1][coluna_pivot] > 0:
                results.append(matriz[i + 1][num_cols - 1] / matriz[i + 1][coluna_pivot])
                linhas.append(i + 1)
        if len(results) > 0:
            valor_pivot = min(results)
            pivot_index = results.index(valor_pivot)
            linha_pivot = linhas[pivot_index]
            valor_pivot = m[linha_pivot][coluna_pivot]
            pivot_encontrado = 1
            return [valor_pivot, linha_pivot, coluna_pivot]
    if pivot_encontrado == 0:
        return -1


def eliminacao_gaussiana(m):
    matriz = np.copy(m)
    if find_pivot(m) == -1:
        return matriz
    else:
        pivot, linha_pivot, col_pivot = find_pivot(matriz)
        matriz[linha_pivot] = np.around(matriz[linha_pivot] / pivot, decimals=6)
        num_linhas = np.shape(matriz)[0]
        for i in range(num_linhas):
            if i != linha_pivot:
                matriz[i] = matriz[i] + (-1) * (matriz[i][col_pivot]) * matriz[linha_pivot]
        return matriz

# realiza a eliminação gaussiana enquanto há elemento negativo, ou enquanto há pivots disponíveis
def simplex(m):
    matriz = np.copy(m)
    num_linhas = matriz.shape[0]
    num_cols = matriz.shape[1]
    while min(matriz[0][num_linhas - 1 : num_cols - 1]) < 0:
        matriz = eliminacao_gaussiana(matriz)
        if find_pivot(matriz) == -1:
            return np.around(matriz, decimals=4)

    return np.around(matriz, decimals=4)


# encontra uma solução viável para o caso ótimo
def find_solution(m):
    matriz = np.copy(m)
    num_cols = matriz.shape[1]
    c = matriz[0][num_rest : num_rest + num_var]
    sol = np.zeros(num_var)
    for i in range(len(c)):
        if c[i] == 0:
            col = matriz[:, i + num_rest]
            pos = np.where(col == 1)[0]
            if len(pos) == 0:
                sol[i] = 0
            else:
                pos = pos[0]
                sol[i] = matriz[pos][num_cols - 1]
    return sol

# checa se a PL é viável por meio da PL auxiliar
def verifica_inviabilidade(A, b, c):
    inviavel = 0
    pl_aux = pl_auxiliar(A, b, c)
    pl_aux_base = base_canonica_pl_auxiliar(pl_aux)
    num_cols = pl_aux.shape[1]
    resultado = simplex(pl_aux_base)
    if resultado[0][num_cols - 1] < 0:
        inviavel = 1
        certificado = resultado[0][:num_rest]
    else:
        certificado = []
    return inviavel, certificado

# checa se a PL é ilimitada (checa se há coluna negativa)
def verifica_ilimitada(m):
    ilimitada = 0
    A = np.copy(m)
    A = A[:,num_rest:-1]
    cols_A = A.shape[1]
    for i in range(cols_A):
        if A[0][i] != 0:
            lista = [A[:,i]<=0]
            if all(lista[0]):
                ilimitada = 1
    return ilimitada

# imprime as respostas para o caso ilimitado
def respostas_ilimitada(m):
    matriz = np.copy(m)
    b = matriz[:,-1]
    matriz = matriz[:,num_rest:-(num_rest+1)]
    num_cols = matriz.shape[1]
    sol = np.zeros(num_var)
    certificado = []
    for i in range(num_cols):
        if matriz[0][i] == 0:
            valores = matriz[:,i]
            index = np.where(valores == 1)[0]
            sol[i] = b[index]
    print('ilimitada')
    print(sol)
    print(certificado)

# imprime as respostas para o caso ótimo
def respostas_otima(m):
    matriz = np.copy(m)
    num_cols = matriz.shape[1]
    certificado = matriz[0][:num_rest]
    valor_opt = matriz[0][num_cols - 1]
    sol = find_solution(matriz)
    print("otima")
    print(valor_opt)
    print(sol)
    print(certificado)


tableau = cria_tableau(A, b, c)
inv = verifica_inviabilidade(A, b, c)
if inv[0] == 1:
    print('inviavel')
    print(inv[1])
else:
    tableau_final = simplex(tableau)
    ilim = verifica_ilimitada(tableau_final)
    if ilim == 1:
        respostas_ilimitada(tableau_final)
    else:
        respostas_otima(tableau_final)
