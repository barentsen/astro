import pymc
import pyfits
import numpy as np
from scipy.interpolate.rbf import Rbf

""" Construct the interpolation functions for the evolutionary model. """
# Load the evolutionary model grid due to Siess et al. (2000), which is stored as a FITS table.
siess = pyfits.getdata("siess_isochrones.fits", 1)
# We use Radial Basis Functions to construct interpolation functions. 
siess_Mr = Rbf(siess.field("logMass"), siess.field("logAge"), siess.field("Mr_iphas"), function="linear")
siess_Mi = Rbf(siess.field("logMass"), siess.field("logAge"), siess.field("Mi_iphas"), function="linear")
siess_Mj = Rbf(siess.field("logMass"), siess.field("logAge"), siess.field("Mj"), function="linear")
siess_logR = Rbf(siess.field("logMass"), siess.field("logAge"), siess.field("logRadius"), function="linear")

""" Construct the interpolation functions for the Halpha colour simulations. """
# Load the grid of simulated colours for H-alpha emission-line objects due to Barentsen et al. (2011)
iphascolours = pyfits.getdata("simulated_iphas_colours_barentsen2011.fits", 1)
# Use Radial Basis Functions to construct interpolation functions.
r_offset = Rbf(iphascolours.field("ri_unred"), iphascolours.field("av"), \
	iphascolours.field("logew"), iphascolours.field("d_r"), function="linear")
ha_offset = Rbf(iphascolours.field("ri_unred"), iphascolours.field("av"), \
	iphascolours.field("logew"), iphascolours.field("d_ha"), function="linear")
i_offset = Rbf(iphascolours.field("ri_unred"), iphascolours.field("av"), \
	iphascolours.field("logew"), iphascolours.field("d_i"), function="linear")
intrinsic = (iphascolours.field("av") == 0) & (iphascolours.field("logew") == -1)
rminHa_intrinsic = Rbf(iphascolours.field("ri_unred")[intrinsic], \
	iphascolours.field("rha")[intrinsic], function="linear")

""" The function make_model() returns all the model variables. """
def make_model(observed_sed, e_observed_sed):
	""" PRIORS """
	
	# Mass prior.
	@pymc.stochastic()
	def logM(value=np.array([np.log10(0.5)]), a=np.log10(0.1), b=np.log10(7)):
		
		def logp(value, a, b):
			# The mass should not fall outside the model limits (a,b).
			if value > b or value < a:
				return -np.Inf
			else:
				# We adopt the Initial Mass Function due to Kroupa (2001).
				mass = 10**value	
				if mass < 0.5: return np.log(mass**-1.3)
				else: return np.log(0.5*mass**-2.3)
				
		def random(a, b):
			val = (b - a) * np.random.rand() + a
			return np.array([val])
	
	# Age prior (logarithmic).
	logT = pymc.Uniform("logT", np.array([5]), np.array([8]))

	# Accretion rate prior (logarithmic).
	logMacc = pymc.Uniform("logMacc", np.array([-15]), np.array([-2]))	
	
	# Disc truncation radius prior.
	Rin = pymc.TruncatedNormal("Rin", mu=np.array([5.0]), tau=2.0**-2.0, a=1.01, b=9e99)
	
	# Distance prior due to Sung (1997).
	d = pymc.TruncatedNormal("d", mu=np.array([760.0]), tau=5.0**-2, a=700, b=9e99)	
	
	# Extinction prior.
	logA0 = pymc.Normal("logA0", mu=np.array([-0.27]), tau=0.46**-2)
	
	
	""" LIKELIHOODS """
	
	# Intrinsic SED likelihood.
	@pymc.deterministic()
	def SED_intrinsic(logM=logM, logT=logT):
		r = siess_Mr(logM, logT)
		i = siess_Mi(logM, logT)
		j = siess_Mj(logM, logT)
		ha = r - rminHa_intrinsic( r-i )
		return np.array([r[0], ha[0], i[0], j[0]])
	
	# Halpha excess luminosity likelihood.
	@pymc.deterministic()
	def logLacc(logM=logM, logT=logT, logMacc=logMacc, Rin=Rin):
		logR = siess_logR(logM, logT)
		return 7.496 + logM + logMacc - logR + np.log10(1 - 1/Rin)
	logLha = pymc.Normal("logLha", mu=(0.64*logLacc - 2.12), tau=0.43**-2)
	
	# H-alpha equivalent width likelihood.
	@pymc.deterministic()
	def logEW(logLha=logLha, SED_intrinsic=SED_intrinsic):
		# Excess and continuum luminosity in the IPHAS Halpha passband.
		Lha = 10**logLha
		Lha_cont = 0.316 * 10**(-0.4*( SED_intrinsic[1] + 0.03 ))
		# Equivalent width.
		ew = -95.0*Lha/Lha_cont
		return np.log10(-ew)
	
	# Apparent SED likelihood.
	@pymc.deterministic()
	def SED_apparent(d=d, logA0=logA0, SED_intrinsic=SED_intrinsic, logEW=logEW):
		# Distance modulus.
		dismod = 5.0*np.log10(d) - 5.0
		# Extinction parameter.
		A0 = 10.**logA0
		# Intrinsic (r'-i').
		ri_intr = np.array([SED_intrinsic[0] - SED_intrinsic[2]])
		# Corrected magnitudes.
		r = SED_intrinsic[0] + dismod + r_offset(ri_intr, A0, logEW)
		ha = SED_intrinsic[1] + dismod + ha_offset(ri_intr, A0, logEW)	
		i = SED_intrinsic[2] + dismod + i_offset(ri_intr, A0, logEW)	
		j = SED_intrinsic[3] + dismod + 0.276*A0
		return np.array([r[0], ha[0], i[0], j[0]])
	
	# Observed SED likelihood.
	@pymc.stochastic(observed=True)
	def SED_observed(value=observed_sed, SED_apparent=SED_apparent):
		e_calib = np.array([0.1, 0.1, 0.1, 0.1])
		D2 = sum( (observed_sed - SED_apparent)**2 / (e_observed_sed**2 + e_calib**2) )
		logp = -D2/2.0
		return logp
	
	# Return all model components.
	return locals()


""" Example code which demonstrates how to sample the joint distribution. """
if __name__ == "__main__":
	# Observed data.
	sed_observed = np.array([19.41, 18.14, 17.56, 15.44]) # [r, Ha, i, J]
	e_sed_observed = np.array([0.03, 0.03, 0.02, 0.06]) # [e_r, e_Ha, e_i, e_J]
	# Initialize the model.
	mymodel = make_model(sed_observed, e_sed_observed)
	M = pymc.MCMC(mymodel)		
	# Run the MCMC algorithm and print the expectation value for log Mass.
	M.sample(50000)
	samples_logM = M.trace("logM")[:]
	print "E[log Mass] = %.2f +/- %.2f" % (np.mean(samples_logM), np.std(samples_logM))


