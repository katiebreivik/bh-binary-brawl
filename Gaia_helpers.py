import numpy as np

def log_g(mass, radius):
    """
        <Computes log_g given mass and radius of the star
        INPUT: Mass in M_sun, radius in R_sun
        OUTPUT: log_g in cm/s^2 >
    """
    G_cgs = 6.67e-8
    Msun_cgs = 1.989e33
    Rsun_cgs = 6.9551e10
    
    g = G_cgs*mass*Msun_cgs/(radius*Rsun_cgs)**2
    
    return np.log10(g)

def M_bol(Lum):
    """
        <Computes absolute magnitude given luminosity in L_sun
        INPUT: luminosity in L_sun 
        OUTPUT: absolute magnitude >
    """
    M_bol = 0.75-2.5*np.log10(Lum/40)
    return M_bol

def m_app(M_bol, dist):
    """
        <Computes apparent magnitude given absolute magnitude and distance 
        INPUT: absolute magnitude and distance in pc 
        OUTPUT: apparent magnitude >
    """
    m_app = M_bol + 5*np.log10(dist/10)
    return m_app



def BC_GaiaG_from_isochrones(teff,logg,feh,bc_grid):
    """
        <Computes BC in the Gaia G-band filter
        INPUT: temperature in K, log_g, metallicity in Fe/H, isochrones bc grid
        OUTPUT: BC>
    """
    Av = [0.0]*len(teff)
    bc = bc_grid.interp([teff, logg, feh, Av], ['G'])           
    
    return(bc)