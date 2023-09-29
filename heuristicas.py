from gradient_descent import Environment
import numpy as np

#Tengo que formalizar que esto es cota minima
def estrellato(env: Environment, Fmax: int) -> int:
    p = env.p_0
    m = env.m
    v = env.v_0
    k=0;
    visto = [False] * 3
    while(True):
        if(visto[0] and visto[1] and visto[2]):
            break
        k+=1
        a = np.array([(-pa)/abs(pa)*Fmax/m for pa in p])
        a[3] -= env.g*m
        pre_p=p
        p+=v*env.dt + a*env.dt**2/2
        v+=a*env.dt
        c=pre_p*p
        if(c[0]<0):
            visto[0]=True
        if(c[1]<0):
            visto[1]=True
        if(c[2]<0):
            visto[2]=True
    return k
    

