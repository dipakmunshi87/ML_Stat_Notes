"""Worked examples of chapter 16: Inference in time: filtering and the Kalman filter

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 16_sequential.py
"""


def example_1():
    """four measurements of a wandering quantity"""
    import numpy as np
    q, r = 1.0, 4.0
    est, P = 0.0, 10.0
    for step, z in enumerate((2.1, 2.9, 2.4, 3.3), 1):
        P += q                                       # predict: the quantity has wandered since the last look
        K = P / (P + r)                              # gain
        est, P = est + K * (z - est), (1 - K) * P    # update
        print("step %d: gain %.2f, estimate %.2f, error bar %.2f" % (step, K, est, np.sqrt(P)))
    P_inf = 0.5 * (-q + np.sqrt(q*q + 4*q*r))
    print("limits: variance %.2f, error bar %.2f, gain %.2f" % (P_inf, np.sqrt(P_inf), P_inf / r))


def example_2():
    """how much to trust the newest measurement"""
    import numpy as np
    q, r = 1.0, 4.0                                  # process and measurement noise variances
    P = q
    for _ in range(100):                             # iterate the filter to its steady state
        P_pred = P + q
        K = P_pred / (P_pred + r)
        P = (1 - K) * P_pred
    print("predicted variance %.2f, gain %.2f, uncertainty after update +/- %.2f, memory %.1f steps" % (P_pred, K, np.sqrt(P), 1 / K))


def example_3():
    """rain or sun? a hidden two-state chain"""
    import numpy as np
    T = np.array([[0.7, 0.3], [0.3, 0.7]])           # transition matrix: rows = today (rain, sun)
    umbrella = np.array([0.9, 0.2])                  # P(umbrella | rain), P(umbrella | sun)
    belief = np.array([0.5, 0.5])
    for day in (1, 2):
        if day > 1:
            belief = belief @ T                      # predict
            print("   predicted P(rain) = %.3f" % belief[0])
        belief = belief * umbrella                   # update on seeing an umbrella
        belief /= belief.sum()
        print("day %d: P(rain) = %.3f" % (day, belief[0]))


if __name__ == "__main__":
    print("\n--- four measurements of a wandering quantity ---")
    example_1()
    print("\n--- how much to trust the newest measurement ---")
    example_2()
    print("\n--- rain or sun? a hidden two-state chain ---")
    example_3()
