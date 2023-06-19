#Importação de bibiotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

#Configuração de janela
np.set_printoptions(linewidth=999999)

#Grau desejado
P = 5

#Domínio Csi
csi0 = -1.                              #Início do domínio
csiP = 1.                               #Fim do domínio
csi = np.linspace(csi0,csiP,num=1000)    #Domínio
csip = np.arange(csi0,csiP+(csiP-csi0)/P,(csiP-csi0)/P) #Pontos do domínio

#Alfa e Beta
a = b = 0.       ##POLINÔMIOS DE LEGENDRE

#Polinômios de Lagrange
def Lagrange(P, csi):
    #Equiespaçado

    #Professor, não consegui achar o motivo pelo qual a
    #base mostra valores maiores que 1. A referência:
    #https://people.sc.fsu.edu/~jburkardt/m_src/lagrange_basis_display/lagrange_basis_display.html
    #Também mostra que a base de Lagrange (equiespaçada) tem valores acima de 1.

    hpeq = []    ##Lista de polinômios
    for p in range(0,P+1):
        hpnum = []        #Polinômios do numerador
        hpden = []        #Termos do denominador
        for q in range(0,P+1):
            if p!=q:
                hpnum.append(csip[q])
                hpden.append(csip[p]-csip[q])
        hpnum = np.poly1d(hpnum,True)
        hpden = np.prod(hpden)
        hpeq.append(hpnum/hpden)

    #Gráfico
    for n in range (0,P+1):
        plt.subplot(2,P+1,n+1)
        hpeqgraf = hpeq[n]
        hpeqgraf = hpeqgraf(csi)
        plt.plot(csi,hpeqgraf)
        plt.axis('tight')
        plt.axis([-1.5, 1.5, hpeqgraf[n].min()-0.5, hpeqgraf[n].max()+0.5])
        plt.xticks([-1, 1])
        plt.yticks([hpeqgraf[n].min(), -1, 1, hpeqgraf[n].max()])
        grau = n
        plt.title('Lagrange h%s (equiespacado)'%grau)
    plt.suptitle('Polinomios de Lagrange (ate grau %s)'%P)

    #Usando Zeros de Gauss-Legendre-Lobato

    hpgll = []
    csiprod = np.poly1d([-1,1],True)       #Polinômio (csi-1)*(csi+1)
    g = csiprod*sp.jacobi(P-1,1,1)
    groots = sorted(np.roots(g))

    for p in range(0,P+1):
        hpnum = []        #Polinômios do numerador
        hpden = []        #Termos do denominador
        for q in range(0,P+1):
            if p!=q:
                hpnum.append(groots[q])
                hpden.append(groots[p]-groots[q])
        hpnum = np.poly1d(hpnum,True)
        hpden = np.prod(hpden)
        hpgll.append(hpnum/hpden)

    for n in range (0,P+1):
        plt.subplot(2,P+1,P+n+1+1)
        hpgllgraf = hpgll[n]
        hpgllgraf = hpgllgraf(csi)
        plt.plot(csi,hpgllgraf)
        plt.axis('tight')
        plt.xticks([-1, 1],fontsize=8)
        plt.axis([-1.5, 1.5, hpgllgraf[n].min()-0.5, hpgllgraf[n].max()+0.5])
        plt.xticks([-1, 1],fontsize=8)
        plt.yticks([-1, 1],fontsize=8)
        grau = n
        plt.title('Lagrange h%s (Gauss-Legendre-Lobato)'%grau, fontsize=8)
    plt.show()
    return

# Jacobi(a,b,P,csi)
Lagrange(P,csi)
