"""Worked examples of chapter 4: Error bars are thermal fluctuations

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 04_fluctuations.py
"""


def example_1():
    """nudging one data point"""
    import numpy as np
    x = np.linspace(-1, 1, 20)
    y = np.array([-0.0578, -0.4708, 0.2324, 0.0531, 0.1977, 0.7224, 0.7013, 0.6845, 1.3723, 0.2087,
                  0.9230, 0.9663, 1.6518, 1.0614, 1.3489, 1.0267, 1.7997, 1.3545, 0.4366, 1.2630])
    sigma, tau = 0.3, 2.0            # noise per point, prior width of each coefficient
    X = np.vander(x, 2, increasing=True)             # columns 1, x
    K = X.T @ X / sigma**2 + np.eye(2) / tau**2      # stiffness = Fisher + prior
    C = np.linalg.inv(K)                             # covariance
    fit = lambda yy: C @ X.T @ yy / sigma**2
    y2 = y.copy(); y2[-1] += 0.3                     # raise the last point by one sigma
    print("best fit", fit(y).round(3), " error bars", np.sqrt(np.diag(C)).round(3))
    print("shift from refitting       ", (fit(y2) - fit(y)).round(4))
    print("covariance times the force ", (C @ X[-1] * 0.3 / sigma**2).round(4))


def example_2():
    """two parameters and a soft mode"""
    import numpy as np
    K = np.array([[10., 8.], [8., 10.]])             # stiffness (Fisher) matrix
    k, modes = np.linalg.eigh(K)
    C = np.linalg.inv(K)
    print("mode stiffnesses", k, " mode sds", (1/np.sqrt(k)).round(3))
    print("marginal sd %.3f, correlation %.1f, conditional sd %.3f" % (np.sqrt(C[0,0]), C[0,1]/C[0,0], 1/np.sqrt(K[0,0])))


def example_3():
    """shot noise"""
    import numpy as np
    N, T = 400, 100.0
    rate = N / T
    fisher = T / rate
    print("rate %.1f per s, Fisher information %.0f, error %.2f, relative error %.0f%%" % (rate, fisher, fisher**-0.5, 100/np.sqrt(N)))


if __name__ == "__main__":
    print("\n--- nudging one data point ---")
    example_1()
    print("\n--- two parameters and a soft mode ---")
    example_2()
    print("\n--- shot noise ---")
    example_3()
