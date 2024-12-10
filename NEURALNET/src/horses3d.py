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
    
    No_of_elements = v1[0]
    Iter = v1[1]
    
    time = np.fromfile(fname, dtype=np.float64, count=1, sep='', offset=144)
    
    ref_values = np.fromfile(fname, dtype=np.float64, count=6, sep='', offset=152)
    
    Mesh = []
    
    offset_value = 152+6*8+4
    

   
    for i in range(0,No_of_elements):
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

        else:
            Ind = 0

        Local_storage.Q = Q1.reshape(size1,order='F')

        Sol[i,:,:,:,:] = Q1[:,:,:,:]

        offset_value = offset_value + size*8

    return Sol

def Q_from_experiment(Name, Nmin, Nmax, NSkip):
    Names = [f"{Name}{i:010d}.hsol" for i in range(Nmin, Nmax, NSkip)]

    Q_list = []

    for file in Names:
        Q1 = Q_from_file(file)
        Q_list.append(Q1)

    return np.stack(Q_list, axis=0)


def Q_SelectEquations(Q_full, Eq):
    return Q_full[:, :, Eq, :, :, :]

def Q_SelectEquations(Q_full, Eq):   

    Q = np.take(Q_full, Eq, axis=2)
    return Q

def Read_experiment(Name, Nmin, Nmax, NSkip, Eq, lo_polynomial, Network_type):
    Q_full = Q_from_experiment(Name, Nmin, Nmax, NSkip)
    Q = Q_SelectEquations(Q_full, Eq)

    N_of_elements = Q.shape[1] 

    if Network_type == "MLP":
        Q_res = Q.reshape(1, -1, np.prod(Q.shape[2:]), order='F')
    elif Network_type == "CNN":
        Q_res = Q.transpose(0, 1, 3, 4, 5, 2).reshape(
            -1, lo_polynomial + 1, lo_polynomial + 1, lo_polynomial + 1, 3
        )
    else:
        raise ValueError(f"Unsupported Network_type: {Network_type}")

    return Q_res, N_of_elements
