import numpy as np
from scipy.interpolate.rbf import Rbf
import pyfits
import pymc

""" Interpolation functions for intrinsic magnitudes """
siess = pyfits.getdata("siess_isochrones.fits", 1)  # Siess et al. (2000)
# Interpolation is performed using linear Radial Basis Functions
siess_Mr = Rbf(siess.field("logMass"), siess.field("logAge"),
               siess.field("Mr_iphas"), function="linear")
siess_Mi = Rbf(siess.field("logMass"), siess.field("logAge"),
               siess.field("Mi_iphas"), function="linear")
siess_Mj = Rbf(siess.field("logMass"), siess.field("logAge"),
               siess.field("Mj"), function="linear")
siess_logR = Rbf(siess.field("logMass"), siess.field("logAge"),
                 siess.field("logRadius"), function="linear")

""" Functions for magnitude offsets due to emission & exctinction """
sim = pyfits.getdata("simulated_iphas_colours_barentsen2011.fits", 1)  # PaperI
# Functions for r'/Ha/i' offsets as a function of colour, extinction and EW
r_offset = Rbf(sim.field("ri_unred"), sim.field("av"), sim.field("logew"),
               sim.field("d_r"), function="linear")
ha_offset = Rbf(sim.field("ri_unred"), sim.field("av"), sim.field("logew"),
                sim.field("d_ha"), function="linear")
i_offset = Rbf(sim.field("ri_unred"), sim.field("av"), sim.field("logew"),
               sim.field("d_i"), function="linear")
# Intrinsic (r'-Ha) colour as a function of intrinsic (r'-i')
intrinsic = (sim.field("av") == 0) & (sim.field("logew") == -1)
rminHa_intrinsic = Rbf(sim.field("ri_unred")[intrinsic],
                       sim.field("rha")[intrinsic], function="linear")


def make_model(observed_sed, e_observed_sed):
    """ This function returns all prior and likelihood objects """

    # Mass prior following the Initial Mass Function due to Kroupa (2001)
    @pymc.stochastic()
    def logM(value=np.array([np.log10(0.5)]), a=np.log10(0.1), b=np.log10(7)):

        def logp(value, a, b):
            if value > b or value < a:
                return -np.Inf  # Stay within the model limits (a,b).
            else:
                mass = 10 ** value
                if mass < 0.5:
                    return np.log(mass ** -1.3)  # Kroupa (2001)
                else:
                    return np.log(0.5 * mass ** -2.3)  # Kroupa (2001)

        def random(a, b):
            val = (b - a) * np.random.rand() + a
            return np.array([val])

    # Age prior (logarithmic).
    logT = pymc.Uniform("logT", np.array([5]), np.array([8]))

    # Accretion rate prior (logarithmic).
    logMacc = pymc.Uniform("logMacc", np.array([-15]), np.array([-2]))

    # Disc truncation radius prior.
    Rin = pymc.TruncatedNormal("Rin", mu=np.array([5.0]), tau=2.0 ** -2,
                               a=1.01, b=9e99)

    # Distance prior due to Sung (1997).
    d = pymc.TruncatedNormal("d", mu=np.array([760.0]), tau=5.0 ** -2,
                             a=700, b=9e99)

    # Extinction prior.
    logA0 = pymc.Normal("logA0", mu=np.array([-0.27]), tau=0.46 ** -2)

    # Intrinsic SED likelihood.
    @pymc.deterministic()
    def SED_intrinsic(logM=logM, logT=logT):
        r = siess_Mr(logM, logT)  # IPHAS r' as a function of (mass, age)
        i = siess_Mi(logM, logT)  # IPHAS i
        j = siess_Mj(logM, logT)  # 2MASS J
        ha = r - rminHa_intrinsic(r - i)  # IPHAS H-alpha
        return np.array([r[0], ha[0], i[0], j[0]])

    # H-alpha excess luminosity likelihood.
    @pymc.deterministic()
    def logLacc(logM=logM, logT=logT, logMacc=logMacc, Rin=Rin):
        logR = siess_logR(logM, logT)  # Radius as a function of (mass, age)
        return 7.496 + logM + logMacc - logR + np.log10(1 - 1 / Rin)
    logLha = pymc.Normal("logLha", mu=(0.64 * logLacc - 2.12), tau=0.43 ** -2)

    # H-alpha equivalent width (EW) likelihood.
    @pymc.deterministic()
    def logEW(logLha=logLha, SED_intrinsic=SED_intrinsic):
        Lha = 10 ** logLha  # Excess luminosity
        Lha_con = 0.316 * 10 ** (-0.4 * (SED_intrinsic[1] + 0.03))  # Continuum
        ew = -95.0 * Lha / Lha_con  # Equivalent width.
        return np.log10(-ew)

    # Apparent SED likelihood.
    @pymc.deterministic()
    def SED_apparent(d=d, logA0=logA0, SED_intr=SED_intrinsic, logEW=logEW):
        dismod = 5.0 * np.log10(d) - 5.0  # Distance modulus.
        A0 = 10.0 ** logA0  # Extinction parameter
        ri_intr = np.array([SED_intr[0] - SED_intr[2]])  # Intrinsic (r'-i')
        # Correct the intrinsic magnitudes for extinction and H-alpha emission:
        r = SED_intr[0] + dismod + r_offset(ri_intr, A0, logEW)
        ha = SED_intr[1] + dismod + ha_offset(ri_intr, A0, logEW)
        i = SED_intr[2] + dismod + i_offset(ri_intr, A0, logEW)
        j = SED_intr[3] + dismod + 0.276 * A0
        return np.array([r[0], ha[0], i[0], j[0]])

    # Observed SED likelihood.
    @pymc.stochastic(observed=True)
    def SED_observed(value=observed_sed, SED_apparent=SED_apparent):
        e_calib = np.array([0.1, 0.1, 0.1, 0.1])  # Absolute uncertainty term
        D2 = sum((observed_sed - SED_apparent) ** 2 /
                 (e_observed_sed ** 2 + e_calib ** 2))
        logp = -D2 / 2.0
        return logp

    # Return all model components.
    return locals()


if __name__ == "__main__":
    """ Example code which demonstrates how to sample the posterior """
    # Input: the observed magnitudes and 1-sigma uncertainties
    sed_observed = np.array([19.41, 18.14, 17.56, 15.44])  # r, Ha, i, J
    e_sed_observed = np.array([0.03, 0.03, 0.02, 0.06])  # e_r, e_Ha, e_i, e_J
    # Initialize the model.
    mymodel = make_model(sed_observed, e_sed_observed)
    M = pymc.MCMC(mymodel)
    # Demo: run the MCMC sampler and print the expectation value for log(Mass)
    M.sample(1000)
    samples_logM = M.trace("logM")[:]
    print "logM = %.2f +/-%.2f" % (np.mean(samples_logM), np.std(samples_logM))
