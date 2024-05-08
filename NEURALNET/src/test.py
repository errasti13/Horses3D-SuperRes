import numpy as np

class storage():
    Q = []

class NNConfig:
    def __init__(self, simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
                 nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model, architecture):
        self.simulation = simulation
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nmin_lo = nmin_lo
        self.nmax_lo = nmax_lo
        self.nskip_lo = nskip_lo
        self.nmin_ho = nmin_ho
        self.nmax_ho = nmax_ho
        self.nskip_ho = nskip_ho
        self.train_percentage = train_percentage
        self.equations = equations
        self.lo_polynomial = lo_polynomial
        self.ho_polynomial = ho_polynomial
        self.trained_model = trained_model
        self.architecture = architecture

def read_nn_config_file(filepath):
    NN_opt = []

    with open(filepath, 'r') as f:
        NN_opt = f.read().splitlines()

    simulation = NN_opt[0].split('\t')[0]
    n_layers = int(NN_opt[1].split('\t')[0])
    n_epochs = int(NN_opt[2].split('\t')[0])
    batch_size = int(NN_opt[3].split('\t')[0])
    nmin_lo = int(NN_opt[4].split('\t')[0])
    nmax_lo = int(NN_opt[5].split('\t')[0])
    nskip_lo = int(NN_opt[6].split('\t')[0])
    nmin_ho = int(NN_opt[7].split('\t')[0])
    nmax_ho = int(NN_opt[8].split('\t')[0])
    nskip_ho = int(NN_opt[9].split('\t')[0])
    train_percentage = float(NN_opt[10].split('\t')[0])
    equations = NN_opt[11].split('\t')[0]
    lo_polynomial = int(NN_opt[12].split('\t')[0])
    ho_polynomial = int(NN_opt[13].split('\t')[0])
    trained_model = NN_opt[14].split('\t')[0]
    architecture = NN_opt[15].split('\t')[0]

    return NNConfig(
        simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
        nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model, architecture
    )

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

def Q_SelectEquations(Q_full,Eq):   
 
    Q = np.zeros(( Q_full.shape[0],Q_full.shape[1],len(Eq),Q_full.shape[3],Q_full.shape[4],Q_full.shape[5] ))
    
    for i in range(0,len(Eq)):
        Q[:,:,i,:,:,:] = Q_full[:,:,Eq[i],:,:,:]
    
    return Q

def Read_experiment(Name, Nmin, Nmax, NSkip, Eq, lo_polynomial, Network_type):
    Q_full = Q_from_experiment(Name, Nmin, Nmax, NSkip)
    Q = Q_SelectEquations(Q_full, Eq)

    N_of_elements = Q.shape[1]

    if Network_type == "MLP":
        Q_res = Q.reshape((1, -1, Q.shape[2] * Q.shape[3] * Q.shape[4] * Q.shape[5]), order='F')
    elif Network_type == "CNN":
        Q = np.transpose(Q, axes=(0, 1, 3, 4, 5, 2))
        Q_res = Q.reshape((-1, lo_polynomial + 1, lo_polynomial + 1, lo_polynomial + 1, 5))
    else:
        print("Network type not implemented")

    return Q_res, N_of_elements

def set_equations(config_nn):
    """
    Sets the equation numbers based on the configuration.

    Args:
        config_nn (NNConfig): The configuration object containing the equation information.

    Returns:
        list: A list of equation numbers based on the configuration.
    """
    equation_sets = {
    'momentum': [2, 3, 4],
    'all': [0, 1, 2, 3, 4]
    }

    return equation_sets.get(config_nn.equations, [])

def main():
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")
    config_nn.equations = 'all'
    Eq = set_equations(config_nn)
    Q_HO_ind, N_of_elements = Read_experiment("RESULTS/TGV_HO_", 20, 21, 1, Eq, config_nn.ho_polynomial, "CNN")
    print(Q_HO_ind.shape)

if __name__ == "__main__":
    main()