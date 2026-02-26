from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from hmm import HMM

import os
os.makedirs("plots", exist_ok=True)

# 1. Reads training text A and test text B
# 2. Trains 2-state HMM for 600 iterations (Baum–Welch / EM)
# 3. Produces plots and analysis for Q1(a)~Q1(d)
# 4. Repeats the same for 4-state HMM (Q2)

# Reads training text A and test text B, converts to integer ids in {0..26}
ALPHABET = list("abcdefghijklmnopqrstuvwxyz") + ["#"]  # 27 symbols
char2id = {ch: i for i, ch in enumerate(ALPHABET)}      # map char -> 0..26

def read_text_to_ids(path: str) -> np.ndarray:
    s = open(path, "r", encoding="utf-8").read()    
    # safety: lowercase     
    s = s.lower()  
    # convert spaces to '#'                                     
    s = s.replace(" ", "#")                              
    # keep only valid symbols (a-z and '#')
    s = "".join(ch for ch in s if ch in char2id)
    # filter unexpected symbols         
    # convert to ids
    return np.array([char2id[ch] for ch in s], dtype=np.int32)  



# 2-state HMM

def init_hmm_2state() -> HMM:
    # number of states， observation symbols
    S = 2                                               
    V = 27                                            

    # initial distribution
    pi = np.array([0.5, 0.5], dtype=float)

    # transitions A[i,j]
    A = np.array([
        [0.49, 0.51],                                   
        [0.51, 0.49],                                 
    ], dtype=float)

    # emissions B[state, symbol] 2*27 matrix
    B = np.zeros((S, V), dtype=float)
    #index for a-m, n-z, and space
    a_to_m = [char2id[ch] for ch in "abcdefghijklm"]   
    n_to_z = [char2id[ch] for ch in "nopqrstuvwxyz"]     
    space = char2id["#"]                                

    # state '0' emission probabilities
    B[0, a_to_m] = 0.0370
    B[0, n_to_z] = 0.0371
    B[0, space] = 0.0367

    # state '1' emission probabilities
    B[1, a_to_m] = 0.0371
    B[1, n_to_z] = 0.0370
    B[1, space] = 0.0367

    # normalization
    B = B / B.sum(axis=1, keepdims=True)

    return HMM(pi=pi, A=A, B=B)


# 4-state HMM 


def init_hmm_4state(seed: int = 0) -> HMM:
    # randomness   different initialization
    rng = np.random.default_rng(seed)                    
    S = 4                                               
    V = 27                                              

    # initial distribution: uniform
    pi = np.ones(S, dtype=float) / S

    # transitions: noice = uniform(-0.01, 0.01)
    A = np.ones((S, S), dtype=float) / S               
    A += rng.uniform(-0.01, 0.01, size=(S, S))           
    A = np.clip(A, 1e-6, None)                           
    A = A / A.sum(axis=1, keepdims=True)              

    # emissions: noise = uniform(-0.005, 0.005) 
    B = np.ones((S, V), dtype=float) / V
    B += rng.uniform(-0.005, 0.005, size=(S, V))
    B = np.clip(B, 1e-6, None)
    B = B / B.sum(axis=1, keepdims=True)

    return HMM(pi=pi, A=A, B=B)


# Training and answering questions

def run_experiment(model: HMM, A_seq: np.ndarray, B_seq: np.ndarray, K: int, tag: str):
    '''
    parameters:
    model: the initialized HMM (2-state or 4-state)
    A_seq：training set A
    B_seq：test set B
    K：number of iterations (600)
    tag：label for output files (e.g., "2state" or "4state
    '''
    # [Q1(b)] Plot the average log-probability of the training and test data after k iterations,
    train_curve = np.zeros(K, dtype=float)
    test_curve = np.zeros(K, dtype=float)

    # [Q1(c)] record emission trajectories for a and n (only for first two states if S>2)
    # Plot the emission probabilities of a few letters for each state as a function k
    # turn letters into ids for indexing
    a_id = char2id["a"]
    n_id = char2id["n"]
    # [Q1(c)] actual frequency of 'a' and 'n' in training text A
    freq_a = np.mean(A_seq == a_id)
    freq_n = np.mean(A_seq == n_id)

    qa = np.zeros((K, model.S), dtype=float)             # store q(a|state s)
    qn = np.zeros((K, model.S), dtype=float)             # store q(n|state s)

    for k in range(K):
        # [Q1(a)] one EM iteration on training set A
        # update the parameters 
        _ = model.baum_welch_one_iter(A_seq, update_pi=False)

        # [Q1(b)] compute average log-prob on train and test
        train_curve[k] = model.avg_logprob(A_seq)
        test_curve[k] = model.avg_logprob(B_seq)

        # [Q1(c)] record emissions for a and n in each state
        # get the emission probabilities of a and n for each state, which is B[state, symbol_id] after the k-th iteration
        qa[k, :] = model.B[:, a_id]
        qn[k, :] = model.B[:, n_id]

        # printing the process
        if (k + 1) % 50 == 0:
            print(f"[{tag}] iter {k+1:3d} | train={train_curve[k]:.6f} | test={test_curve[k]:.6f}")

    # [Q1(b)] Plot the average log-probability curves for training and test sets
    plt.figure()
    plt.plot(np.arange(1, K + 1), train_curve, label="train avg log P(A)/|A|")
    plt.plot(np.arange(1, K + 1), test_curve, label="test  avg log P(B)/|B|")
    plt.xlabel("iteration k")
    plt.ylabel(r"Average Log-Likelihood $\frac{1}{|O|}\log P(O)$")
    plt.title(f"{tag}: average log-probability (Q1b)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{tag}_Qb_logprob.png"), dpi=200)


    # [Q1(c)] Plot emission probability trajectories
    plt.figure()
    for s in range(model.S):
        plt.plot(np.arange(1, K + 1), qa[:, s], label=f"q(a|state{s+1})")
    for s in range(model.S):
        plt.plot(np.arange(1, K + 1), qn[:, s], label=f"q(n|state{s+1})", linestyle="--")
    # using dotted and dashed lines to distinguish a and n
    plt.axhline(y=freq_a, color="gray", linestyle=":", linewidth=2,
            label="freq(a) in A")

    plt.axhline(y=freq_n, color="gray", linestyle="--", linewidth=2,
            label="freq(n) in A")
    plt.xlabel("iteration k")
    plt.ylabel("emission probability")
    plt.title(f"{tag}: emission evolution (Q1c)")
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{tag}_Qc_emissions_a_n.png"), dpi=200)

    # [Q(d)] Analysis: interpret what the HMM learned
    print(f"\n[{tag}] Q(d) Parameter interpretation")
    # most different letters
    if model.S == 2:
        # For 2-state: direct absolute difference |q(.|1) - q(.|2)|
        diff = np.abs(model.B[0] - model.B[1]) 
        #sort by difference in descending order
        order = np.argsort(-diff)
        print("\nTop 10 symbols with largest |q(.|1)-q(.|2)|:")
        # print the symbol, the difference, and the emission probabilities in both states   
        for idx in order[:10]:
            print(
                f"symbol='{ALPHABET[idx]}' "
                f"diff={diff[idx]:.6f}  "
                f"q1={model.B[0, idx]:.6f}  "
                f"q2={model.B[1, idx]:.6f}"
            )

    else:
        # 4-state use max-min spread across states
        # calculate the spread of emission probabilities for each symbol across states, which is max_s q(.|s) - min_s q(.|s)
        # find the max and min emission probabilities for each symbol across states (every column )
        spread = model.B.max(axis=0) - model.B.min(axis=0)
        #sort by spread in descending order
        order = np.argsort(-spread)

        print("\nTop 10 symbols with largest (max_s q(.|s) - min_s q(.|s)) across states:")
        for idx in order[:10]:
            # which state gives max/min for this symbol
            s_max = int(np.argmax(model.B[:, idx]))
            s_min = int(np.argmin(model.B[:, idx]))
            # print the symbol, the spread, and the max/min emission probabilities and their corresponding states
            print(
                f"  symbol='{ALPHABET[idx]}' "
                f"spread={spread[idx]:.6f}  "
                f"max=state{s_max+1}:{model.B[s_max, idx]:.6f}  "
                f"min=state{s_min+1}:{model.B[s_min, idx]:.6f}"
            )
    # print the transition matrix A
    print("\nTransition matrix A (rows: from-state, cols: to-state):")
    print(model.A)
    # save the plots
    print("\nSaved plots:")
    print(f"  - {tag}_Q1b_logprob.png")
    print(f"  - {tag}_Q1c_emissions_a_n.png")
    print("=" * 60 + "\n")

'''
def main():
    A_seq = read_text_to_ids("data/textA-1.txt")
    B_seq = read_text_to_ids("data/textB-1.txt")

    print("Length of training A:", len(A_seq))
    print("Length of test B:", len(B_seq))
    # number of iterations for EM training
    K = 600  
    # 2 state experiment
    model2 = init_hmm_2state()
    run_experiment(model2, A_seq, B_seq, K=K, tag="2state")
    # 4 state experiment
    model4 = init_hmm_4state(seed=0)
    run_experiment(model4, A_seq, B_seq, K=K, tag="4state")
'''

def main():
    A_seq = read_text_to_ids("data/textA-1.txt")
    B_seq = read_text_to_ids("data/textB-1.txt")

    print("Length of training A:", len(A_seq))
    print("Length of test B:", len(B_seq))

    K = 600

    model2 = init_hmm_2state()
    run_experiment(model2, A_seq, B_seq, K=K, tag="2state")

    model4 = init_hmm_4state(seed=0)
    run_experiment(model4, A_seq, B_seq, K=K, tag="4state")

    return model2, model4  

if __name__ == "__main__":
    main()



'''
Length of training A: 30000
Length of test B: 5000
[2state] iter  50 | train=-2.856794 | test=-2.836020
[2state] iter 100 | train=-2.856794 | test=-2.836020
[2state] iter 150 | train=-2.856794 | test=-2.836019
[2state] iter 200 | train=-2.856792 | test=-2.836018
[2state] iter 250 | train=-2.856733 | test=-2.835960
[2state] iter 300 | train=-2.754599 | test=-2.734784
[2state] iter 350 | train=-2.753890 | test=-2.733226
[2state] iter 400 | train=-2.753877 | test=-2.733163
[2state] iter 450 | train=-2.753874 | test=-2.733150
[2state] iter 500 | train=-2.753873 | test=-2.733148
[2state] iter 550 | train=-2.753873 | test=-2.733148
[2state] iter 600 | train=-2.753873 | test=-2.733149

[2state] Q(d) Parameter interpretation

Top 10 symbols with largest |q(.|1)-q(.|2)|:
symbol='#' diff=0.343947  q1=0.000004  q2=0.343951
symbol='e' diff=0.213953  q1=0.000000  q2=0.213953
symbol='t' diff=0.137293  q1=0.145278  q2=0.007985
symbol='a' diff=0.132541  q1=0.002687  q2=0.135228
symbol='o' diff=0.132515  q1=0.000000  q2=0.132515
symbol='i' diff=0.117980  q1=0.000000  q2=0.117980
symbol='s' diff=0.110296  q1=0.110296  q2=0.000000
symbol='n' diff=0.106303  q1=0.106303  q2=0.000000
symbol='r' diff=0.104732  q1=0.104732  q2=0.000000
symbol='l' diff=0.076781  q1=0.076781  q2=0.000000

Transition matrix A (rows: from-state, cols: to-state):
[[0.28630266 0.71369734]
 [0.74052404 0.25947596]]

Saved plots:
  - 2state_Q1b_logprob.png
  - 2state_Q1c_emissions_a_n.png
============================================================

[4state] iter  50 | train=-2.856781 | test=-2.836006
[4state] iter 100 | train=-2.855149 | test=-2.834565
[4state] iter 150 | train=-2.594352 | test=-2.597200
[4state] iter 200 | train=-2.593014 | test=-2.596810
[4state] iter 250 | train=-2.584096 | test=-2.592164
[4state] iter 300 | train=-2.583354 | test=-2.592605
[4state] iter 350 | train=-2.583248 | test=-2.592987
[4state] iter 400 | train=-2.583237 | test=-2.593067
[4state] iter 450 | train=-2.583237 | test=-2.593069
[4state] iter 500 | train=-2.583237 | test=-2.593069
[4state] iter 550 | train=-2.583237 | test=-2.593069
[4state] iter 600 | train=-2.583237 | test=-2.593069

[4state] Q(d) Parameter interpretation

Top 10 symbols with largest (max_s q(.|s) - min_s q(.|s)) across states:
  symbol='#' spread=0.906894  max=state4:0.906894  min=state2:0.000000
  symbol='e' spread=0.228369  max=state2:0.228369  min=state4:0.000000
  symbol='a' spread=0.226530  max=state2:0.226530  min=state3:0.000000
  symbol='o' spread=0.223320  max=state2:0.223320  min=state4:0.000000
  symbol='t' spread=0.213184  max=state3:0.213184  min=state2:0.000000
  symbol='n' spread=0.165948  max=state1:0.165948  min=state4:0.000000
  symbol='i' spread=0.159909  max=state2:0.159909  min=state3:0.000000
  symbol='s' spread=0.129913  max=state1:0.129913  min=state4:0.000000
  symbol='r' spread=0.109582  max=state1:0.109582  min=state4:0.000000
  symbol='c' spread=0.096951  max=state3:0.096951  min=state2:0.000000

Transition matrix A (rows: from-state, cols: to-state):
[[3.32871824e-01 8.10899276e-03 1.07037357e-01 5.51981827e-01]
 [6.56293940e-01 9.96629358e-02 1.55614244e-01 8.84288807e-02]
 [8.40033405e-03 8.83842619e-01 1.04985589e-01 2.77145743e-03]
 [1.41368515e-10 2.88451049e-01 7.11548951e-01 1.00000000e-12]]

Saved plots:
  - 4state_Q1b_logprob.png
  - 4state_Q1c_emissions_a_n.png


'''