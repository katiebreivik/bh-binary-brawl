"""A collection of methods to generate synthetic photometry and other observables"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
#from dustmaps.bayestar import BayestarQuery
import mwdust
combined19= mwdust.Combined19()
from schwimmbad import MultiPool
import tqdm
def log_g(mass, radius):
    """ Computes log g in cgs units
    Parameters
    ----------
    mass : `float/array`
        mass in solar masses
    radius : `float/array`
        radius in solar radii
    Returns
    -------
    log g : `float/array`
        log surface gravity in cgs
    """
    G_cgs = 6.67e-8
    Msun_cgs = 1.989e33
    Rsun_cgs = 6.9551e10
    
    g = G_cgs*mass*Msun_cgs/(radius*Rsun_cgs)**2
    
    return np.log10(g)

def M_absolute_bol(lum):
    """Computues the absolute bolometric luminosity
    Parameters
    ----------
    lum : `float/array`
        luminosity in solar luminosities
    
    Returns
    -------
    M_bol : `float/array`
        absolute bolometric magnitude
    """
    log_lum = np.log10(lum)
    M_bol = 4.75-2.7*log_lum
    return M_bol

def m_apparent(M_abs, dist):
    #distance in parsecs
    m_app = M_abs + 5*np.log10(dist/10)
    return m_app

def m_abs(m_app, dist):
    #distance in parsecs
    M_abs = m_app - 5*np.log10(dist/10)
    return M_abs


def get_mags(lum, distance, teff, logg, Fe_h, Av, bc_grid, filters):
    """ uses isochrones bolometric correction method to interpolate 
    across the MIST bolometric correction grid
    
     Parameters
    ----------
    lum : `array`
        luminosity in Lsun
    
    distance : `array`
        distance in kpc
    
    teff : `array`
        effective temperature in K
    
    logg : `array`
        log g in cgs
    
    Fe_h : `array`
        metallicity
    
    Av : `array`
        extinction correction 
    
    bc_grid : `isochrones bolometric correction grid object`
        object which generates bolometric corrections!
        
    Returns
    -------
    mags : `list of arrays`
        list of apparent magnitude arrays that matches the filters provided 
        prepended with the bolometric apparent magnitude
    """
    M_abs = M_absolute_bol(lum=lum)
    m_app = m_apparent(M_abs=M_abs, dist=distance * 1000)
    BCs_abs = bc_grid.interp([teff, logg, Fe_h, Av], filters)
    
    BCs_app = bc_grid.interp([teff, logg, Fe_h, Av], filters)
        
    mags_app = []
    mags_app.append(m_app)
    for ii, filt in zip(range(len(filters)), filters):
        mags_app.append(m_app - BCs_app[:,ii])
    mags_abs = []
    mags_abs.append(m_app)
    for ii, filt in zip(range(len(filters)), filters):
        mags_abs.append(M_abs - BCs_abs[:,ii])
    
    return mags_app, mags_abs

def addMags(mag1, mag2):
    """ Adds two stellar magnitudes
    Parameters
    ----------
    mag1, mag2 : `float/array`
        two magnitudes from two stars
    Returns
    -------
    magsum : `float/array`
        returns the sum of mag1 and mag2
    """
    magsum = -2.5*np.log10(10**(-mag1*0.4)+10**(-mag2*0.4))
    return magsum


def get_EB_V(dat):
    l, b, D = dat
    EB_V = combined19(l, b, D)
    return EB_V
    

def get_extinction(dat):
    """Calculates the visual extinction values from the dat
    DataFrame using the dustmaps. bayestar query
    Parameters
    ----------
    dat : `DataFrame`
        contains Galactocentric cartesian coordinates
        with names [units]: X [kpc], Y [kpc], Z [kpc]
    Returns
    -------
    Av : `array`
        Visual extinction values for all points in dat
    """
    c = SkyCoord(x=np.array(dat.X) * u.kpc,
                 y=np.array(dat.Y) * u.kpc, 
                 z=np.array(dat.Z) * u.kpc,
                 frame=coord.Galactocentric,
                 galcen_distance = 8.5 * u.kpc,
                 z_sun = 36.0e-3 * u.kpc)
    
    galactic = c.galactic
    
    l = np.arange(min(galactic.l.value), max(galactic.l.value) + 5, 5)
    b = np.arange(min(galactic.b.value), max(galactic.b.value) + 1, 1)

    EB_V = np.zeros(len(galactic))
    for l_hi, l_low in tqdm.tqdm(zip(l[1:], l[:-1]), total=len(l[1:])):
        l_mid = (l_hi - l_low)/2
        for b_hi, b_low in zip(b[1:], b[:-1]):
            b_mid = (b_hi - b_low)/2
            ind, = np.where((galactic.l.value < l_hi) & (galactic.l.value >= l_low) & 
                            (galactic.b.value < b_hi) & (galactic.b.value >= b_low))
            if len(ind) > 0:            
                EB_V[ind] = combined19(l_mid + l_low, b_mid + b_low, galactic.distance.value[ind])
    Av = 3.2 * EB_V
    
    return Av  


def get_photometry_1(dat, bc_grid):
    # Now let's check out the brightness of the companions in 2MASS filters
    # for this we need to calculate log g of the companion
    dat['logg_1'] = log_g(dat.mass_1, dat.rad_1)
    
    mags_app, mags_abs = get_mags(lum = dat.lum_1.values, 
                                  distance = dat.dist.values, 
                                  teff = dat.teff_1.values, 
                                  logg = dat.logg_1.values, 
                                  Fe_h = dat.FeH.values, 
                                  Av = dat.Av.values, 
                                  bc_grid = bc_grid,
                                  filters = ['J', 'H', 'K', 'G', 'BP', 'RP'])
    
    [m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, BP_app_1, RP_app_1] = mags_app 
    [m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1, BP_abs_1, RP_abs_1] = mags_abs
    
    
    return m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, BP_app_1, RP_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1, BP_abs_1, RP_abs_1
    
def get_photometry_2(dat, bc_grid):
    # Now let's check out the brightness of the companions in 2MASS filters
    # for this we need to calculate log g of the companion
    dat['logg_2'] = log_g(dat.mass_2, dat.rad_2)
    
    mags_app, mags_abs = get_mags(lum = dat.lum_2.values, 
                                  distance = dat.dist.values, 
                                  teff = dat.teff_2.values, 
                                  logg = dat.logg_2.values, 
                                  Fe_h = dat.FeH.values, 
                                  Av = dat.Av.values, 
                                  bc_grid = bc_grid,
                                  filters = ['J', 'H', 'K', 'G', 'BP', 'RP'])
    
    [m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, BP_app_2, RP_app_2] = mags_app 
    [m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2, BP_abs_2, RP_abs_2] = mags_abs
    
    
    return m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, BP_app_2, RP_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2, BP_abs_2, RP_abs_2


def get_phot(sim_set, sys_type, bc_grid):
    """Computes J,H,K photometry subject to dust extinction
    using the MIST boloemtric correction grid which contains
    filters J,H,K
    Parameters
    ----------
    sim_set : `DataFrame`
        dataset of cosmic binaries at present dat
    sys_type : `int`
        system type; choose from:
        singles = 0; binaries = 1; bh binaries = 2; 
    bc_grid : `MISTBolometricCorrectionGrid`
        bolometric correction grid which works with isochrones
        to compute BCs
    Returns
    -------
    sim_set : `DataFrame`
        dataset of cosmic binaries at present dat with added 
        photometry and extinction information
    """
    sim_set['Av'] = get_extinction(sim_set)
    print('pop size before extinction cut: {}'.format(len(sim_set)))
    #sim_set.loc[sim_set.Av > 6, ['Av']] = 6
    #sim_set = sim_set.fillna(6)
    #print('pop size after extinction cut: {}'.format(len(sim_set)))
    
    if sys_type == 0:
        phot_1 = get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, BP_app_1, RP_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1, BP_abs_1, RP_abs_1 = phot_1          
        
        sim_set['mbol_app'] = m_app_1
        sim_set['J_app'] = J_app_1
        sim_set['H_app'] = H_app_1
        sim_set['K_app'] = K_app_1
        sim_set['G_app'] = G_app_1
        sim_set['BP_app'] = BP_app_1
        sim_set['RP_app'] = RP_app_1
        
        sim_set['mbol_abs'] = m_abs_1
        sim_set['J_abs'] = J_abs_1
        sim_set['H_abs'] = H_abs_1
        sim_set['K_abs'] = K_abs_1
        sim_set['G_abs'] = G_abs_1
        sim_set['BP_abs'] = BP_abs_1
        sim_set['RP_abs'] = RP_abs_1
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_1.values
        sim_set['teff_obs'] = sim_set.teff_1.values

    elif sys_type == 1:
        phot_1 = get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, BP_app_1, RP_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1, BP_abs_1, RP_abs_1 = phot_1  
        
        phot_2 = get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, BP_app_2, RP_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2, BP_abs_2, RP_abs_2 = phot_2
                
        # check if the primary or secondary is brighter in 2MASS K
        sys_bright = np.ones(len(sim_set))
        
        # next handle the systems where there was merger and the leftover star
        # is left in kstar_2 instead of kstar_1
        kstar_1 = sim_set.kstar_1.values
        ind_single_1 = np.where(kstar_1 == 15)[0]
        sys_bright[ind_single_1] = 2.0
        
        # next; in some instances, there are systems which are too dim to register
        # in the isochrones/MIST grids
        ind_dim_1 = np.where(np.isnan(G_app_1))[0]
        sys_bright[ind_dim_1] = 2.0
        ind_dim_2 = np.where(np.isnan(G_app_2))[0]
        #ind_dim_2 already covered above
        
        ind_2_bright = np.where(G_app_2 < G_app_1)[0]
        ind_1_bright = np.where(G_app_2 >= G_app_1)[0]
        sys_bright[ind_2_bright] = 2.0
        #ind_1_bright already covered above
        
        sim_set['sys_bright'] = sys_bright
        
        logg_obs = np.zeros(len(sim_set))
        logg_obs[sys_bright == 1.0] = sim_set.loc[sim_set.sys_bright == 1].logg_1
        logg_obs[sys_bright == 2.0] = sim_set.loc[sim_set.sys_bright == 2].logg_2
        sim_set['logg_obs'] = logg_obs
        
        teff_obs = np.zeros(len(sim_set))
        teff_obs[sys_bright == 1.0] = sim_set.loc[sim_set.sys_bright == 1].teff_1
        teff_obs[sys_bright == 2.0] = sim_set.loc[sim_set.sys_bright == 2].teff_2
        sim_set['teff_obs'] = teff_obs
        
        
        sim_set['J_app'] = addMags(J_app_1, J_app_2)
        sim_set['H_app'] = addMags(H_app_1, H_app_2)
        sim_set['K_app'] = addMags(K_app_1, K_app_2)
        sim_set['G_app'] = addMags(G_app_1, G_app_2)
        sim_set['BP_app'] = addMags(BP_app_1, BP_app_2)
        sim_set['RP_app'] = addMags(RP_app_1, RP_app_2)
        sim_set['mbol_app'] = addMags(m_app_1, m_app_2)
    
        sim_set['J_abs'] = addMags(J_abs_1, J_abs_2)
        sim_set['H_abs'] = addMags(H_abs_1, H_abs_2)
        sim_set['K_abs'] = addMags(K_abs_1, K_abs_2)
        sim_set['G_abs'] = addMags(G_abs_1, G_abs_2)
        sim_set['BP_abs'] = addMags(BP_abs_1, BP_abs_2)
        sim_set['RP_abs'] = addMags(RP_abs_1, RP_abs_2)
        sim_set['mbol_abs'] = addMags(m_abs_1, m_abs_2)
    
    elif sys_type == 2:
        phot_2 = get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, BP_app_2, RP_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2, BP_abs_2, RP_abs_2 = phot_2
        
        sim_set['mbol_app'] = m_app_2
        sim_set['J_app'] = J_app_2
        sim_set['H_app'] = H_app_2
        sim_set['K_app'] = K_app_2
        sim_set['G_app'] = G_app_2
        sim_set['BP_app'] = BP_app_2
        sim_set['RP_app'] = RP_app_2
        
        sim_set['mbol_abs'] = m_abs_2
        sim_set['J_abs'] = J_abs_2
        sim_set['H_abs'] = H_abs_2
        sim_set['K_abs'] = K_abs_2
        sim_set['G_abs'] = G_abs_2
        sim_set['BP_abs'] = BP_abs_2
        sim_set['RP_abs'] = RP_abs_2
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = 2*np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_2.values
        sim_set['teff_obs'] = sim_set.teff_2.values
        
    return sim_set