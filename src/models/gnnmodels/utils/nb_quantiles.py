def nb_quantiles(mu, alpha, quantiles):
    """
    Exact NB2 quantiles via scipy, as a float numpy array [..., Q].

    Parameterisation: size r = 1/alpha, success probability p = r / (r + mu),
    so mean = mu and variance = mu + alpha * mu^2.
    """
    import numpy as np
    from scipy.stats import nbinom

    mu    = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    alpha = np.clip(np.asarray(alpha, dtype=float), 1e-10, None)
    r     = 1.0 / alpha
    p     = r / (r + mu)

    q = np.asarray(quantiles, dtype=float)
    return nbinom.ppf(q, r[..., None], p[..., None]).astype(float)

