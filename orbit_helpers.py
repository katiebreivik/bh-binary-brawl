import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import astropy.units as u

def random_orientation(dat):
    dat['inc'] = np.random.choice([-1, 1], len(dat)) * np.arccos(np.random.uniform(0, 1, len(dat)))
    dat['OMEGA'] = np.random.uniform(0, 2 * np.pi, len(dat))
    dat['omega'] = np.random.uniform(0, 2 * np.pi, len(dat))
    
    return dat

def get_TI_constants(m1, m2, sep, porb, ecc, inc, OMEGA, omega):
    """
        <Computes sky projected orbital size of a binary orbit 
        INPUT: m1,m2 in M_sun, orbital separation in AU, orbital period in years,
                    ecc, inclination(ins), argument of periapses(omega) and longitude of ascending node(OMEGA) in radians
        OUTPUT: Thiele Innes constants >
    """
    A_list = []
    B_list = []
    F_list = []
    G_list = []
    X_list = []
    Y_list = []
    for jj in range(len(m1)):
        # solve the binary for x,y positions
        
        
        E = keplerEqSolve(m1[jj],m2[jj],ecc[jj],porb[jj],0,porb[jj])

        
        A = sep[jj]*(np.cos(omega[jj])*np.cos(OMEGA[jj])-np.sin(omega[jj])*np.sin(OMEGA[jj])*np.cos(inc[jj]))
        B = sep[jj]*(np.cos(omega[jj])*np.sin(OMEGA[jj])+np.sin(omega[jj])*np.cos(OMEGA[jj])*np.cos(inc[jj]))
    
        F = sep[jj]*(-np.sin(omega[jj])*np.cos(OMEGA[jj])-np.cos(omega[jj])*np.sin(OMEGA[jj])*np.cos(inc[jj]))
        G = sep[jj]*(-np.sin(omega[jj])*np.sin(OMEGA[jj])+np.cos(omega[jj])*np.cos(OMEGA[jj])*np.cos(inc[jj]))
        
        X=np.cos(E)-ecc[jj]
        Y=(1-ecc[jj]**2)**0.5*np.sin(E)
        
        A_list.append(A)
        B_list.append(B)
        F_list.append(F)
        G_list.append(G)
        X_list.append(X)
        Y_list.append(Y)
        
    return A_list, B_list, F_list, G_list, X_list, Y_list


def orbitProject(m1, m2, sep, porb, ecc, inc, OMEGA, omega):
    """
        <Computes sky projected orbital size of a binary orbit 
        INPUT: m1,m2 in M_sun, orbital separation in AU, orbital period in years,
                    ecc, inclination(ins), argument of periapses(omega) and longitude of ascending node(OMEGA) in radians
        OUTPUT: orbital separation in AU >
    """
    distProj = []
    
    for jj in range(len(m1)):
        # solve the binary for x,y positions
        
        
        E = keplerEqSolve(m1[jj],m2[jj],ecc[jj],porb[jj],0,porb[jj])

        
        A = sep[jj]*(np.cos(omega[jj])*np.cos(OMEGA[jj])-np.sin(omega[jj])*np.sin(OMEGA[jj])*np.cos(inc[jj]))
        B = sep[jj]*(np.cos(omega[jj])*np.sin(OMEGA[jj])+np.sin(omega[jj])*np.cos(OMEGA[jj])*np.cos(inc[jj]))
    
        F = sep[jj]*(-np.sin(omega[jj])*np.cos(OMEGA[jj])-np.cos(omega[jj])*np.sin(OMEGA[jj])*np.cos(inc[jj]))
        G = sep[jj]*(-np.sin(omega[jj])*np.sin(OMEGA[jj])+np.cos(omega[jj])*np.cos(OMEGA[jj])*np.cos(inc[jj]))
        
        X=np.cos(E)-ecc[jj]
        Y=(1-ecc[jj]**2)**0.5*np.sin(E)
    
        xProj = A*X + F*Y
        yProj = B*X + G*Y
        
        distMax, coord1, coord2 = maxDist(xProj,yProj)
        
        distProj.append(distMax)
        
    return np.array(distProj)


def maxDist(xProject, yProject):
    """
        <Computes two farthest points on the orbit
        INPUT: x and y coordinate of the locus of orbit
        OUTPUT: farthest pints and distance between the points>
    """
    distMax = 0
    for d1 in zip(xProject, yProject):
        for d2 in zip(xProject, yProject):
            dist = distance(d1,d2)
            if dist>distMax:
                distMax = dist
                save1 = d1
                save2 = d2
            
    return distMax, save1, save2


def distance(p0, p1):
    """
        <Computes euclidian distance between two given points 
        INPUT: points p1 nad p2
        OUTPUT: euclidian distance >
    """
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def keplerEqSolve(m1,m2,e,Porb,tmin,tmax):
    Msun = 1.9891 * 10**30

    ### NOTE HERE THAT WE REQUIRE Porb & tEvolve to have same units!!
    theta = []    
    E = []
    m1 = m1*Msun
    m2 = m2*Msun
    nStep = 50
    tRange = np.linspace(tmin,tmax,nStep)
    # print('datatype = ',type(Porb))
    for time in tRange:
        nHalfPorb = int(2*(time-1)/Porb)
        PsiDiff = 1
        M = 2*np.pi*time/Porb
        PsiOld = M
        theta0old = 180.0
        while PsiDiff > 1e-10:
        #print PsiDiff, PsiOld, e*np.sin(PsiOld)
            PsiNew = M + e*np.sin(PsiOld)
            PsiDiff = PsiNew-PsiOld
            PsiOld = PsiNew
        theta0 = 2*np.arctan(((1+e)/(1-e))**(0.5)*np.tan(PsiOld/2.))
        theta.append(theta0)  
        E.append(PsiOld)
    return np.array(E)


def sep_from_porb(m1, m2, porb):
    """
        <Computes orbital separation
        INPUT: mass of binary components(m1,m2) in M_sun,
                orbital period in years 
        OUTPUT: absolute magnitude >
    """
    sep = ((m1 + m2)*porb**2)**(1./3.)
    return sep


def delta_ql(q, l):
    """
    Computes the mass and luminosity ratio factors as in eq (6) of Belokurov+20
    
    Params
    ------
    q : numpy.array
        mass ratio where q = m2/m1
    
    l : numpy.array
        luminosity ratio where l = lum2/lum1
        
    Returns
    -------
    d_ql : numpy.array
        mass and luminosity ratio factor
    """
    d_ql = np.abs(q-l)/((q+1) * (l+1))
    
    return d_ql


def delta_ql_bright(m1, m2, l1, l2):
    # the delta ql is defined such that l=L_BH/L_LC is always less than one
    # for BH + bright stars, the secondary is the bright object, so q = M_BH/M_LC
    
    q = np.where(l1 > l2, m1/m2, m2/m1)
    l = np.where(l1 > l2, l1/l2, l2/l1)
    dql = delta_ql(q, l)
    
    return dql

def get_delta_ql_co(m1, m2, l1, l2):
    
    # the delta ql is defined such that l=L_BH/L_LC is always less than one
    # for BH + bright stars, the secondary is the bright object, so q = M_BH/M_LC
    q = m1/m2
    l = np.zeros_like(q)

    dql = delta_ql(q, l)
    
    
    return dql


def get_projected_orbit(dat, bintype='normie'):
    rsun_in_au = 215.0954
    day_in_year = 365.242

    
    dat['sep_project'] = orbitProject(
        m1=dat.mass_1.values, m2=dat.mass_2.values, 
        sep=dat.sep.values / rsun_in_au, porb=dat.porb.values / day_in_year, 
        ecc=dat.ecc.values, inc=dat.inc.values, 
        OMEGA=dat.OMEGA.values, omega=dat.omega.values)
    if bintype == 'normie':
        dat['dql'] = delta_ql_bright(
            m1=dat.mass_1.values, m2=dat.mass_2.values, 
            l1=dat.lum_1.values, l2=dat.lum_2.values)
        dat['sep_phot'] = dat.sep_project.values * (1 - dat.dql.values)
    elif bintype == 'co':
        dat['dql'] = get_delta_ql_co(
            m1=dat.mass_1.values, m2=dat.mass_2.values, 
            l1=dat.lum_1.values, l2=dat.lum_2.values)
        dat['sep_phot'] = dat.sep_project.values * (1 - dat.dql.values)
        
    else:
        print('you need to specify bintype of normie or co')
    
    return dat

    
def parallax(dist):
    """
        <Computes parallax
        INPUT: distance in parsec 
        OUTPUT: parallax in arc sec >
    """
    return (1/dist)


def get_delta_theta(dat):
    sep = dat.sep.values * u.Rsun
    m_bright = np.where(dat.mass_1 > dat.mass_2, dat.mass_2.values, dat.mass_1.values) * u.Msun
    m_dim = np.where(dat.mass_1 > dat.mass_2, dat.mass_1.values, dat.mass_2.values) * u.Msun
    dist = dat.dist.values * u.kpc
    dat['delta_theta'] = ((sep.to(u.AU).value)/(1 + m_bright/m_dim)) / dist.to(u.kpc).value
    return dat


def get_theta(dat):
    sep = dat.sep.values * u.Rsun
    m_bright = np.where(dat.mass_1 > dat.mass_2, dat.mass_2.values, dat.mass_1.values) * u.Msun
    m_dim = np.where(dat.mass_1 > dat.mass_2, dat.mass_1.values, dat.mass_2.values) * u.Msun
    dist = dat.dist.values * u.kpc
    dat['theta'] = ((sep.to(u.AU).value)) / dist.to(u.kpc).value/1000
    return dat

def get_sigma_AL(G):
    Gmag =  [5, 5.5, 6, 6.38, 6.57, 6.72, 6.85, 7.03, 7.27, 7.4, 7.77, 8, 8.4,  8.5, 9,
             9.5, 10, 10.6, 11, 11.25, 11.5, 12, 12.4, 12.7, 12.9, 13.1, 13.5, 14, 15, 
             16, 17, 18, 19, 20, 21]

    Sigma_al = [0.42, 0.27, 0.19, 0.192, 0.2, 0.19, 0.19, 0.192, 0.2, 0.205, 0.215, 
                0.215, 0.205, 0.2,0.163, 0.157, 0.156, 0.147, 0.173, 0.19, 0.176, 
                0.145,0.135, 0.145, 0.16,0.135, 0.143, 0.169, 0.249, 0.39, 0.65, 
                1.165,2.21,4.65,9.5]
    
    sigma_interp = interp1d(Gmag, Sigma_al, fill_value = [9.5])
    
    return sigma_interp(G)

def get_rho(dat):
    sigma_AL = get_sigma_AL(dat.G_app)
    
    dat['rho'] = np.sqrt(dat.delta_theta**2/sigma_AL**2 + 1)
    
    return dat