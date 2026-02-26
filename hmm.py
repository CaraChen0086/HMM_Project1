import numpy as np  # numerical computing

'''
homework 02 : this is the model, what is the probability to generate 0110.
project: what should my model look like to understand the text 0110 the best?
'''

'''
Math Fundamentals 
'''

#Compute log(sum(exp(vec)))
def logsumexp(vec: np.ndarray) -> float:
    m = np.max(vec)                 
    # if all are -inf, sum is 0
    # log(0) = -inf
    if np.isneginf(m):                    
        return -np.inf                       
    return m + np.log(np.sum(np.exp(vec - m)))  
    # x_i-m ≤ 0, avoid overflow/underflow

#Normalize each row to sum to 1
def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.clip(mat, eps, 1.0)                 
    # avoid exact 0 which might cause log(-inf)
    mat = mat / mat.sum(axis=1, keepdims=True)   
    # normalization
    return mat

'''
HMM class: parameters computation and EM updates
'''

#forward /backward + EM updates
class HMM:
    """
      S = number of hidden states
      V = number of observation symbols (27: a-z,'#')
      pi[s] = initial probability of state s
      A[i,j] = transition probability of i -> j
      B[j,v] = emission probability of symbol v in state j
      
      cache parameters and their log-space

    """
    # input initial distribution, transitions, emissions
    # 2-state / 4-state
    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray):
        # initial distribution
        self.pi = pi.astype(float)      
        # transitions     
        self.A = A.astype(float)         
        # emissions     
        self.B = B.astype(float)            
        # ensure valid probabilities and cache log values 
        # log-space calculation 
        self._renormalize_and_cache_logs()    


    #read the dimensions of the HMM parameters : eg: 2 states or 4 states
    @property
    # return number of states, which is the number of rows in A
    def S(self) -> int:
        return int(self.A.shape[0])         

    @property
    # return vocabulary size, which is the number of columns in B
    def V(self) -> int:
        return int(self.B.shape[1])           

    def _renormalize_and_cache_logs(self) -> None:
        """
        Ensure parameters pi/A/B are valid distributions and cache their log versions.
        """
        #avoid zeros which cause log(-inf)
        eps = 1e-12                                          
        self.pi = np.clip(self.pi, eps, 1.0)      
        #normalize pi         
        self.pi = self.pi / self.pi.sum()          
        #normalize A and B rows
        self.A = normalize_rows(self.A, eps=eps)         
        self.B = normalize_rows(self.B, eps=eps)              
        # cache logs for emission and transition probabilities
        self.log_pi = np.log(self.pi)                   
        self.log_A = np.log(self.A)                           
        self.log_B = np.log(self.B)                     


    '''
    calculate E-step: froward and backward 
    first step of EM: using the current parameters to compute log_alpha, log_beta, and loglikelihood,posterior
    '''
    def forward_log(self, O: np.ndarray) -> tuple[np.ndarray, float]:
        """
         The forward probability at time t and state s is the joint probability of observing the first t observations and being in state s at time t
        define Log-space forward algorithm.
        αt​(s)

        Using the current model parameters (π, A, B), to compute:
        The forward probabilities for each state at every time step.
        The log-likelihood of the entire observation sequence.

        """
        # T is the sequence length
        T = len(O)                  
        # init with -inf                       
        log_alpha = np.full((T, self.S), -np.inf)           

        # Initializatio
        log_alpha[0, :] = self.log_pi + self.log_B[:, O[0]]

        # Recursion since t=1
        # log_alpha[t,j] = log_B[j,O[t]] + logsum_i (log_alpha[t-1,i] + log_A[i,j])
        # calculate log_alpha[t,j] for each t and j
        for t in range(1, T):
            for j in range(self.S):
                log_alpha[t, j] = self.log_B[j, O[t]] + logsumexp(log_alpha[t - 1, :] + self.log_A[:, j])

        # Termination: log P(O) = logsum_s log_alpha[T-1, s]
        loglik = logsumexp(log_alpha[T - 1, :])

        return log_alpha, loglik

    def backward_log(self, O: np.ndarray) -> np.ndarray:
        """
        The probability of observing the remaining observations given that we are in state s at time t.
        Log-space backward algorithm.
        βt​(s)
        Input: O : observation sequence of ints, shape (T,)
        Outputs: log_beta: log backward table, shape (T,S)

        """
        # length 
        T = len(O)                      
        # init with -inf                   
        log_beta = np.full((T, self.S), -np.inf)             

        # Initialization: log_beta[T-1, s] = log(1) = 0
        log_beta[T - 1, :] = 0.0

        # Recursion: from t=T-2 down to 0
        # log_beta[t,i] = logsum_j (log_A[i,j] + log_B[j,O[t+1]] + log_beta[t+1,j])
        for t in range(T - 2, -1, -1):
            for i in range(self.S):
                log_beta[t, i] = logsumexp(self.log_A[i, :] + self.log_B[:, O[t + 1]] + log_beta[t + 1, :])

        return log_beta


    '''
    one EM update
    homework 3
    using the current parameters to compute the expected counts (gamma and xi) and then update the parameters A and B using these expected counts.
    '''
    # Baum–Welch (EM) update 
    def baum_welch_one_iter(self, O: np.ndarray, update_pi: bool = False) -> float:
        """
        1. E-step：Expectation Step: using forward/backward to compute the posterior posterior（γ、ξ）
        2. M-step: Maximization Step: using the posterior（γ、ξ） as "soft counts" to update A and B
    
        """
        # length of the observation sequence, and number of time steps 
        T = len(O)                                           
        # E-step forward/backward : compute log_alpha, log_beta, and loglik  
        # forward in log-space
        log_alpha, loglik = self.forward_log(O)         
        # backward in log-space
        log_beta = self.backward_log(O)                     
        # Compute gamma[t,s] = P(state_t=s | O) 
        # gamma is the posterior probability of being in state s at time t 
        # γt​(s)： γt​(s)∝αt​(s)⋅βt​(s) 
        # In log domain: log_gamma[t,s] ∝ log_alpha[t,s] + log_beta[t,s]
        # unnormalized log posterior
        log_gamma = log_alpha + log_beta     
        # normalize each t so sum_s gamma[t,s]=1                
        for t in range(T):
            log_gamma[t, :] -= logsumexp(log_gamma[t, :])    
        # convert back to probability domain for counting
        gamma = np.exp(log_gamma)                           

        # Compute xi[t,i,j] 
        # log_xi[t,i,j] ∝ log_alpha[t,i] + log_A[i,j] + log_B[j,O[t+1]] + log_beta[t+1,j]
        # xi is the soft count expected count of transitions from i to j at time t, given the whole observation sequence O
        # xi[t,i,j] = P∗{ti​=t}
        xi = np.zeros((T - 1, self.S, self.S), dtype=float)  
        for t in range(T - 1):
            # ξt​(i,j)∝αt​(i)Aij​Bj​(Ot+1​)βt+1​(j)
            log_xi_t = (log_alpha[t, :, None]               # at time t, and state i
                        + self.log_A                         # probability of transition i->j
                        + self.log_B[:, O[t + 1]][None, :]   # the emission probability of symbol O[t+1] from state j
                        + log_beta[t + 1, :][None, :])       # the backward probability of being in state j at time t+1 and observing the rest of the sequence

            # normalize xi_t so sum_{i,j} xi_t(i,j)=1
            log_xi_t -= logsumexp(log_xi_t.reshape(-1))      # subtract log normalizer
            #convert back to probability domain
            xi[t] = np.exp(log_xi_t)                       


        # M-step: update parameters using expected counts from gamma and xi

        # pi is the expected count of starting in state s, which is gamma[0,s]
        if update_pi:
            self.pi = gamma[0].copy()                       

        # Equation 37 : Update A: A[i,j] = sum_t xi[t,i,j] / sum_t gamma[t,i] 
        # soft count: expected transitions i->j, shape (S,S) 
        # xi sum = c∗(t)
        A_num = xi.sum(axis=0)                     
        # all the expected outgoing from i, shape (S,1)          
        A_den = gamma[:-1].sum(axis=0)[:, None]              
        self.A = A_num / np.clip(A_den, 1e-300, None)        

        # Equation 36:Update B: B[s,v] = sum_{t:O[t]=v} gamma[t,s] / sum_t gamma[t,s]
        B_num = np.zeros((self.S, self.V), dtype=float)      # numerator counts
        # sum gamma[t,s] over those positions
        # the expected count of emitting symbol v from state s is the sum of gamma[t,s] for all t where O[t]=v
        for v in range(self.V):                              
            mask = (O == v)                                
            if np.any(mask):
                B_num[:, v] = gamma[mask].sum(axis=0)        
        # all expected emissions from state s, shape (S,1)
        B_den = gamma.sum(axis=0)[:, None]                 
        self.B = B_num / np.clip(B_den, 1e-300, None)        

        # Re-normalize and refresh logs
        self._renormalize_and_cache_logs()

        return loglik

    '''
    evaluation tool 
    '''
    #return average log-probability
    def avg_logprob(self, O: np.ndarray) -> float:
        # compute log-likelihood
        _, loglik = self.forward_log(O)                      
        # average per character
        return loglik / len(O)                               
