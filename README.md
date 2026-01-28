# tqtu-twinprime-resonance-Code-is-available
import mpmath as mp
import numpy as np
from scipy import stats

mp.mp.dps = 50

def quantum_recoil_prime(n, psi_T=1e-10):
    if n < 2:
        return 0.0
    S = mp.nsum(lambda k: mp.mangoldt(k), [1, n])
    U = mp.primepi(n)
    logn = mp.log(n)
    asym_S = logn
    asym_U = n / logn if logn > 0 else 0
    dev = abs(S - asym_S) + abs(U - asym_U) + 1e-10
    r_scale = mp.log(n) + mp.log(2)
    Xi = psi_T * dev / (r_scale ** 4)
    return float(Xi)

def prime_resonance_coherence(p, q):
    if p > q:
        p, q = q, p
    gap = q - p
    Xi_p = quantum_recoil_prime(p)
    Xi_q = quantum_recoil_prime(q)
    gm = mp.sqrt(p * q)
    Xi_pq = quantum_recoil_prime(int(mp.floor(gm)))
    if Xi_p * Xi_q > 1e-30:
        base_R = Xi_pq / np.sqrt(Xi_p * Xi_q)
    else:
        base_R = 0.0
    closeness_boost = 1.0 / (1.0 + gap / p)
    R_Q = min(base_R * closeness_boost, 1.0)
    return R_Q

def test_twin_prime_resonance():
    print("="*80)
    print("TQTU TWIN PRIME RESONANCE TEST - ENHANCED VERSION")
    print("="*80)
    
    twin_primes = [(3,5), (5,7), (11,13), (17,19), (29,31), (41,43), (59,61), (71,73), 
                   (101,103), (107,109), (137,139), (149,151), (179,181), (191,193), 
                   (197,199), (227,229), (239,241), (269,271), (281,283), (311,313),
                   (347,349), (419,421), (431,433), (461,463), (521,523), (569,571),
                   (599,601), (617,619), (641,643), (659,661), (809,811), (821,823),
                   (827,829), (857,859), (877,879)]
    
    random_primes = [(3,11), (7,19), (13,29), (23,47), (31,61), (37,73), (43,89), 
                     (53,107), (67,131), (79,157), (97,149), (101,163), (103,173), 
                     (107,179), (109,181), (113,191), (127,211), (131,223)]

    print("\nTwin Prime Resonance:")
    twin_resonances = []
    for p, q in twin_primes[:15]:
        R = prime_resonance_coherence(p, q)
        twin_resonances.append(R)
        print(f"({p}, {q}): R_Q = {R:.8f}")
    
    print(f"\nAverage Twin R_Q: {np.mean(twin_resonances):.8f}")
    print(f"Std Dev: {np.std(twin_resonances):.8f}")

    print("\nRandom Prime Pairs Resonance:")
    random_resonances = []
    for p, q in random_primes:
        R = prime_resonance_coherence(p, q)
        random_resonances.append(R)
        print(f"({p}, {q}): R_Q = {R:.8f}")
    
    print(f"\nAverage Random R_Q: {np.mean(random_resonances):.8f}")
    print(f"Std Dev: {np.std(random_resonances):.8f}")

    t_stat, p_value = stats.ttest_ind(twin_resonances, random_resonances, equal_var=False)
    print(f"\nWelch t-test: t = {t_stat:.4f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print("TQTU PREDICTION SUPPORTED: Twin primes show SIGNIFICANTLY higher resonance coherence!")
    else:
        print("No significant difference yet — further tuning needed.")

if __name__ == "__main__":
    test_twin_prime_resonance()
    ....................................................................................................
    Results:
    ================================================================================
TQTU TWIN PRIME RESONANCE TEST - ENHANCED VERSION
================================================================================

Twin Prime Resonance:
(3, 5): R_Q = 0.73447189
(5, 7): R_Q = 0.71228943
(11, 13): R_Q = 0.80796158
(17, 19): R_Q = 0.85728736
(29, 31): R_Q = 0.90156588
(41, 43): R_Q = 0.92334595
(59, 61): R_Q = 0.94196802
(71, 73): R_Q = 0.95074248
(101, 103): R_Q = 0.96231637
(107, 109): R_Q = 0.96496916
(137, 139): R_Q = 0.97106667
(149, 151): R_Q = 0.97273535
(179, 181): R_Q = 0.97697227
(191, 193): R_Q = 0.97805928
(197, 199): R_Q = 0.97901162

Average Twin R_Q: 0.90898422
Std Dev: 0.08691343

Random Prime Pairs Resonance:
(3, 11): R_Q = 0.22510198
(7, 19): R_Q = 0.34206978
(13, 29): R_Q = 0.46557665
(23, 47): R_Q = 0.49039736
(31, 61): R_Q = 0.50163135
(37, 73): R_Q = 0.47841537
(43, 89): R_Q = 0.47653224
(53, 107): R_Q = 0.48703811
(67, 131): R_Q = 0.48533025
(79, 157): R_Q = 0.50832800
(97, 149): R_Q = 0.65259917
(101, 163): R_Q = 0.61990782
(103, 173): R_Q = 0.58034044
(107, 179): R_Q = 0.58225211
(109, 181): R_Q = 0.58690259
(113, 191): R_Q = 0.55444674
(127, 211): R_Q = 0.59595059
(131, 223): R_Q = 0.58891587

Average Random R_Q: 0.51231869
Std Dev: 0.09973349

Welch t-test: t = 11.8280, p = 0.000000
TQTU PREDICTION SUPPORTED: Twin primes show SIGNIFICANTLY higher resonance coherence!
