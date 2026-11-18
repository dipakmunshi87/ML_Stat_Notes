"""Worked examples of chapter 12: Eigenmodes: Karhunen-Loève, PCA and signal-to-noise

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 12_eigenmodes.py
"""


def example_1():
    """two correlated measurements"""
    import numpy as np
    C = np.array([[2.0, 1.2], [1.2, 1.0]])
    lam, vec = np.linalg.eigh(C)
    print("eigenvalues", lam.round(2), " largest carries %.0f%%" % (100 * lam[-1] / lam.sum()))
    print("patterns", np.abs(vec[:, 1]).round(2), np.abs(vec[:, 0]).round(2), " sds", np.sqrt(lam[::-1]).round(2))


def example_2():
    """a ring of four points"""
    import numpy as np
    from scipy.linalg import circulant
    C = circulant([1.0, 0.5, 0.2, 0.5])              # correlation depends only on separation
    print("eigenvalues of the covariance:", np.sort(np.linalg.eigvalsh(C))[::-1].round(2))
    print("Fourier transform of the row: ", np.sort(np.fft.fft(C[0]).real)[::-1].round(2))


def example_3():
    """which combination of two pixels to keep"""
    import numpy as np
    from scipy.linalg import eigh
    S = np.array([[1.0, 0.9], [0.9, 1.0]])           # signal covariance
    N = np.diag([0.1, 1.0])                          # noise covariance
    lam = eigh(S, N, eigvals_only=True)[::-1]        # signal-to-noise eigenvalues
    info = 0.5 * (lam / (1 + lam))**2                # Fisher information on the amplitude, per mode
    print("signal-to-noise", lam.round(2), " information", info.round(3), " share of the first: %.0f%%" % (100 * info[0] / info.sum()))
    print("equal noise 0.5:", (np.linalg.eigvalsh(S)[::-1] / 0.5).round(1))


if __name__ == "__main__":
    print("\n--- two correlated measurements ---")
    example_1()
    print("\n--- a ring of four points ---")
    example_2()
    print("\n--- which combination of two pixels to keep ---")
    example_3()
