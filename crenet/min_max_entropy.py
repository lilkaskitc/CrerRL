from scipy.optimize import linprog
from scipy.optimize import minimize
import numpy as np


def search_values(l, u, delta=1e-5):
    n = len(l)
    c = np.zeros(n)
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if res.success:
        return res.x
    else:
        return None

def min_max_entropy(l, u, x0, delta=1e-5):
    n = len(l)
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]
    # from scipy.optimize import maximize
    # Objective function
    def objective_min(x):
        return -np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    def objective_max(x):
        return np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    # Constraint function
    def constraint(x):
        return np.sum(x) - 1

    problem = {'type': 'eq', 'fun': constraint}
    sol1 = minimize(objective_min, x0, method='SLSQP', bounds=bounds, constraints=problem)
    sol2 = minimize(objective_max, x0, method='SLSQP', bounds=bounds, constraints=problem)
    return sol1.fun, -sol2.fun


def min_max_entropy_calculation(pred):
    """
    Arguments:
    ModeType: 1-Total uncertainty (entropy); 2-Epistemic uncertainty (entropy)
    3-Epistemic uncertainty (range of the probability interval)

    Returns:
    prediction uncertainty
    """
    # Number of classes
    Nc = int(pred.shape[-1] / 2)

    # Extract lower and upper probabilities
    if type(pred) == np.ndarray:
        predL = pred[:, :Nc]
        predU = pred[:, Nc:]
    else:
        predL = pred[:, :Nc].numpy()
        predU = pred[:, Nc:].numpy()

    # Compute the uncertainty measures
    mins = list()
    maxs = list()
    for i in range(predL.shape[0]):
        x0 = search_values(predL[i, :], predU[i, :])
        if x0 is None:
            print(str(i), ": No valid solution found.")
        else:
           min, max = min_max_entropy(predL[i, :], predU[i, :], x0, delta=1e-5)
           if min is None or max is None:
              print(str(i), ": No valid solution found.")
           else:
              mins.append(min)
              maxs.append(max)
    u_pred = {
        'Hu': np.array(maxs),
        'EU': np.array(maxs) - np.array(mins),
        'Hl': np.array(mins)
    }
    return u_pred