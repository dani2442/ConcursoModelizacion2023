from gradient_descent import Environment
import numpy as np

# Acotación de impacto
def estrellato(env: Environment, Fmax: int) -> int:
    p = np.copy(env.p_0)
    m = env.m
    v = np.copy(env.v_0)
    a = np.array([(-pa)/abs(pa)*Fmax/m for pa in p])
    a[2] -= env.g*m
    k=0
    visto = [False] * 3
    while(True):
        if(visto[0] and visto[1] and visto[2]):
            break
        k+=1
        pre_p=np.copy(p)
        p+=v*env.dt + a*env.dt**2/2
        v+=a*env.dt
        c=[pre_p[0]*p[0],pre_p[1]*p[1],pre_p[2]*p[2]]
        if(c[0]<=0):
            visto[0]=True
        if(c[1]<=0):
            visto[1]=True
        if(c[2]<=0):
            visto[2]=True
    return k
    

