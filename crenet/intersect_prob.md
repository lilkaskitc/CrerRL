To implement the calculations for the **Credal-Set Interval Neural Networks (CreINNs)**, we will focus on the three core equations:

1. 
**Equation 8**: Reachable probability interval calculation (redundancy reduction).


2. 
**Equation 12**: Intersection probability calculation for a specific class.


3. 
**Equation 13**: Calculation of the constant $\alpha$ used in the intersection probability transform.



### Python Implementation

The following code assumes you have a NumPy array of lower probabilities `q_lower` and upper probabilities `q_upper` for each class (derived from Equation 7 in the paper).

```python
import numpy as np

def calculate_creinn_probabilities(q_lower, q_upper):
    """
    Implements Equations 8, 12, and 13 from the CreINN paper.
    
    Args:
        q_lower (np.ndarray): Array of lower probability bounds for each class.
        q_upper (np.ndarray): Array of upper probability bounds for each class.
        
    Returns:
        tuple: (q_reachable_lower, q_reachable_upper, q_int)
    """
    # Ensure inputs are numpy arrays for vectorization
    q_l = np.array(q_lower)
    q_u = np.array(q_upper)
    num_classes = len(q_l)
    
    # --- Equation 8: Reachable Probability Intervals ---
    # Calculates the tightened, reachable bounds for the credal set.
    # q_star_lower_k = max(q_lower_k, 1 - sum_{j!=k} q_upper_j)
    # q_star_upper_k = min(q_upper_k, 1 - sum_{j!=k} q_lower_j)
    
    q_star_lower = np.zeros(num_classes)
    q_star_upper = np.zeros(num_classes)
    
    for k in range(num_classes):
        # Sum of upper/lower bounds excluding the current class k
        sum_upper_jk = np.sum(q_u) - q_u[k]
        sum_lower_jk = np.sum(q_l) - q_l[k]
        
        [cite_start]q_star_lower[k] = max(q_l[k], 1 - sum_upper_jk) [cite: 122]
        [cite_start]q_star_upper[k] = min(q_u[k], 1 - sum_lower_jk) [cite: 122]

    # --- Equation 13: Calculate Alpha ---
    # Alpha determines how to distribute the remaining probability mass.
    # alpha = (1 - sum(q_star_lower)) / sum(q_star_upper - q_star_lower)
    
    sum_q_star_lower = np.sum(q_star_lower)
    sum_diff = np.sum(q_star_upper - q_star_lower)
    
    # Avoid division by zero if intervals are point-wise
    if sum_diff == 0:
        alpha = 0.0
    else:
        [cite_start]alpha = (1 - sum_q_star_lower) / sum_diff [cite: 125]

    # --- Equation 12: Intersection Probability (q_int) ---
    # Computes the single probability distribution for class prediction.
    # q_int_k = q_star_lower_k + alpha * (q_star_upper_k - q_star_lower_k)
    
    [cite_start]q_int = q_star_lower + alpha * (q_star_upper - q_star_lower) [cite: 124, 125]
    
    return q_star_lower, q_star_upper, q_int

# Example Usage:
# Assume 3 classes with probability intervals
ql = [0.1, 0.2, 0.1]
qu = [0.5, 0.6, 0.4]

reachable_l, reachable_u, q_final = calculate_creinn_probabilities(ql, qu)

print(f"Reachable Lower: {reachable_l}")
print(f"Reachable Upper: {reachable_u}")
print(f"Intersection Prob (q_int): {q_final}")
print(f"Sum of q_int: {np.sum(q_final)}") # Should be 1.0

```

### Explanation of the Steps

* 
**Equation 8 (Reachable Bounds):** This step reduces redundancy in the initial probability intervals. It ensures that for every class $k$, there exists at least one valid probability distribution within the credal set where the probability is exactly equal to the lower or upper bound.


* 
**Equation 13 ($\alpha$):** This constant represents a weight between 0 and 1. It is used to normalize the intervals into a single probability vector that sums to 1.


* 
**Equation 12 ($q_{int}$):** This "intersection probability transform" equally weights the interval for each class to provide a unique representative probability distribution. This vector is then typically used with `argmax` to make the final class prediction.