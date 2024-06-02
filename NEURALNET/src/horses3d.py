import numpy as np

class storage():
    Q = []

def Q_from_file_2(fname):  
    v1 = np.fromfile(fname, dtype=np.int32, count=2, sep='', offset=136)
    
    No_of_elements = v1[0]
    Iter = v1[1]
    
    time = np.fromfile(fname, dtype=np.float64, count=1, sep='', offset=144)
    
    ref_values = np.fromfile(fname, dtype=np.float64, count=6, sep='', offset=152)
    
    Mesh = []
    
    offset_value = 152+6*8+4
    
    
      
    for i in range(0,No_of_elements):

        Local_storage = storage     

        # offset_value = offset_value + 4
        # P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        # offset_value = offset_value + 4*4   
        # size = P_order[0]*P_order[1]*P_order[2]*P_order[3]
        

            
        # Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
            
        
        
        offset_value = offset_value + 4
        P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        offset_value = offset_value + 4*4   
        size = P_order[0]*P_order[1]*P_order[2]*P_order[3]   
        
        Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        offset_value = offset_value + size*8
        #Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        if (i==0):
            Sol = np.zeros((No_of_elements,P_order[0],P_order[1],P_order[2],P_order[3] ))
        else:
            continue
        
        Sol[i,:,:,:,:] =  np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order[0],P_order[1],P_order[2],P_order[3],order='F')
        
   
    return Sol


def Q_from_file(fname):
    v1 = np.fromfile(fname, dtype=np.int32, count=2, sep='', offset=136)
    #print(v1)
    
    No_of_elements = v1[0]
    #print(No_of_elements)
    Iter = v1[1]
    
    time = np.fromfile(fname, dtype=np.float64, count=1, sep='', offset=144)
    
    ref_values = np.fromfile(fname, dtype=np.float64, count=6, sep='', offset=152)
    
    Mesh = []
    
    offset_value = 152+6*8+4
    

   
    for i in range(0,No_of_elements):
        
        #print(i)
        Ind = 0
        
        Local_storage = storage     

        offset_value = offset_value + 4
        P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        offset_value = offset_value + 4*4   
        size = P_order[0]*P_order[1]*P_order[2]*P_order[3]
        

            
        Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        Q1 = np.zeros( (Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3] )   )
        size1 = P_order[0]*P_order[1]*P_order[2]*P_order[3] 
        Q1 =Q

        if (i==0):
            Sol = np.zeros((No_of_elements,P_order[0],P_order[1],P_order[2],P_order[3]))
            #Sol = np.zeros((No_of_elements,size1))
        else:
            Ind = 0
        #print("Q:")
        #print(Q)
        #Local_storage.Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        Local_storage.Q = Q1.reshape(size1,order='F')
        #print("L Q:")
        #print(Local_storage.Q)
        # if (i==1):
        #     print("Aqui",Q1[1,0,0,0])
        # else:
        #     Ind = 0
        Sol[i,:,:,:,:] = Q1[:,:,:,:]
        #for j in range(0,size1):
        #    Sol[i,j]=Local_storage.Q[j]
        # 'F' en el reshape significa que lo haga como en FORTRAN
        offset_value = offset_value + size*8
        
        #Mesh.append(Local_storage)
    return Sol

def Q_from_experiment(Name,Nmin,Nmax,NSkip):
    Names = []
    
    for i in range(Nmin,Nmax,NSkip):
        s1 = f'{i:010d}'
        Names.append(Name+s1+".hsol") 
       
    N = len(Names)
    Q1   = 0

    for i in range(1,N+1):
        Q1 = Q_from_file(Names[i-1])
        if (i==1):             
            Q = np.zeros((N,Q1.shape[0],Q1.shape[1],Q1.shape[2],Q1.shape[3],Q1.shape[4]) )
            Q[i-1,:,:,:,:,:] = Q1[:,:,:,:,:] 
        else:
            Q[i-1,:,:,:,:,:] = Q1[:,:,:,:,:]  
        
           
    return Q

def Q_SelectEquations(Q_full, Eq):   

    Q = np.take(Q_full, Eq, axis=2)
    return Q

def Read_experiment(Name, Nmin, Nmax, NSkip, Eq, lo_polynomial, Network_type):
    Q_full = Q_from_experiment(Name, Nmin, Nmax, NSkip)
    Q = Q_SelectEquations(Q_full, Eq)

    N_of_elements = Q.shape[1]

    if Network_type == "MLP":
        Q_res = Q.reshape((1, -1, Q.shape[2] * Q.shape[3] * Q.shape[4] * Q.shape[5]), order='F')
    elif Network_type == "CNN":
        Q = np.transpose(Q, axes=(0, 1, 3, 4, 5, 2))
        Q_res = Q.reshape((-1, lo_polynomial + 1, lo_polynomial + 1, lo_polynomial + 1, 3))
    else:
        print("Network type not implemented")

    return Q_res, N_of_elements