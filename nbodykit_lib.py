
# py39 environment (Python 3.9)
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py, scipy, logging

import nbodykit as nb # Installed in my pypelid environment

def open_fits_file(infile):
    with fits.open(infile) as hdus:
        hdus.info()
        hdus.readall()
        primary = hdus[0]
        header = hdus[1].header
        data  = hdus[1].data
    galaxies = Table(data).to_pandas()
    #display(galaxies.describe())
    #if header: return (galaxies, header)
    #else: 
    return galaxies


def convert_hdf5_to_dataframe(pypfile, unmeasured_mask=True):
    # Ok I'm caving and writing an automated way to get a DataFrame out of the Pypelid HDF5 file
    with h5py.File(pypfile,'r') as f:
        blockns = np.array(f['data']) 
        #print(blockns)   
        #print(f['data/0'].attrs.keys())
        #print(f['data/0'].attrs.get('count'))
        columns = np.array(f['data/'+blockns[0]])
        #print(columns)

        ngals = 0
        for i in blockns:
            n = f['data/'+i].attrs.get('count')
            print(n)
            ngals+=n
        print("Number of galaxies in file =",ngals)

        # This will take a LONG time
        blocks = []
        for i in blockns:
            block_dict = {}
            for col in columns:
                if 'coord' in col:
                    block_dict['RA'] = pd.Series(f['data/'+i+'/'+col][:,0])
                    block_dict['Dec'] = pd.Series(f['data/'+i+'/'+col][:,1])
                else: 
                    block_dict[col] = pd.Series(f['data/'+i+'/'+col])

            blocks.append(pd.DataFrame(block_dict))

    # Now merge
    galsout = pd.concat(blocks)
    del blocks

    print(len(galsout))
    if unmeasured_mask:
        galsout = galsout[galsout['zmeas']>-1].rename(\
                columns={'RA':'ra_gal','Dec':'dec_gal','zmeas':'observed_redshift_gal'})
    else: 
        galsout = galsout.rename(\
                columns={'RA':'ra_gal','Dec':'dec_gal','zmeas':'observed_redshift_gal'})
    ngo = len(galsout)
    print('Measured galaxies =', ngo)

    return galsout


def convert_hdf5_to_fits_save(infile=None, outfile=None, overwrite=False):

    galsout = convert_hdf5_to_dataframe(infile)

    # First make a correctly formatted input fits file from the Pypelid output 
    # 1: Open a new hdu with astropy.io.fits
    #display(galsout.describe())
    #hdu = fits.BinTableHDU(Table.from_pandas(galsout[['dec_gal','ra_gal','observed_redshift_gal', 'z', 'id']]))
    hdu = fits.BinTableHDU(Table.from_pandas(galsout))
    #hdu.header['COMMENT'] = 'By Dida Markovic with Pypelid on Flagship in May 2022'
    hdu.header['COMMENT'] = 'By Sujeong Lee with Pypelid on Flagship in Oct 2022'

    # 2: Make sure the header makes sense
    #    https://docs.astropy.org/en/stable/io/fits/usage/headers.html
    #display(hdu.header)

    # 3: Save the 3 relevant columns of galsout dataframe to hdu
    hdu.writeto(outfile, overwrite=overwrite)
    print ('save fits to', outfile)


def save_dataframe_to_fits_save(dataframe=None, filename=None, overwrite=False):

    #galsout = covert_hdf5_to_dataframe(infile)

    # First make a correctly formatted input fits file from the Pypelid output 
    # 1: Open a new hdu with astropy.io.fits
    #display(galsout.describe())
    hdu = fits.BinTableHDU(Table.from_pandas(dataframe))
    #hdu.header['COMMENT'] = 'By Dida Markovic with Pypelid on Flagship in May 2022'
    #hdu.header['COMMENT'] = 'By Sujeong Lee with Pypelid on Flagship in Oct 2022'

    # 2: Make sure the header makes sense
    #    https://docs.astropy.org/en/stable/io/fits/usage/headers.html
    #display(hdu.header)

    # 3: Save the 3 relevant columns of galsout dataframe to hdu
    hdu.writeto(filename, overwrite=overwrite)
    print ('save fits to', filename)


def construct_randoms(galaxies, ramin=None, ramax=None, decmin = None, decmax=None,
 factor=10, filename=None, window=None, center=None):
    import globygon.catalog as gb # Should be installable from Pypy (so pip install it)

    degtorad = np.pi/180.

    # generate 50xrandoms
    ngo = galaxies.shape[0]

    NRO = ngo*factor
    print('multi factor=', factor)
    print('NRO =',NRO)
    #    B & C: Draw random RA & Dec pairs on the sky
    # Need to do this gradually, because 400M x 3 coordinates can't be held in memory (50 GB)
    # First draw random positions in a cube.
    dn = int(NRO/10) # steps of how many randoms at a time (note that this is how many we try - we must select the circle still)
    nr = 0
    RA_list = []; Dec_list = []
    while nr < NRO:

        # create randoms 
        RA, Dec = create_random_on_sphere( ramin=ramin, ramax=ramax, decmin=decmin, decmax=decmax, size=dn )

        # Cut out the circle
        _ = gb.Catalog(RA,Dec)
        rmask = _.calculate_dist_from_com(point=center) < window
        RA = RA[rmask]
        Dec = Dec[rmask]

        nr+=len(RA)
        RA_list.append(RA)
        Dec_list.append(Dec)
        del RA,Dec # Try to release memory
        print('{:>10d}'.format(nr),flush=True)


    # Make a new view (I think) as long arrays
    all_RA = np.concatenate(RA_list)
    all_Dec = np.concatenate(Dec_list)
    print(all_RA.shape,all_Dec.shape)


    # 3: B: Assign post-pypelid n(z) (18 minutes) OR C: Assign original n(z)
    # Now sample out nr galaxies and assign the random coordinates in the circle selected above
    outrand = galaxies.sample(nr,random_state=1,replace=True)
    outrand['ra_gal'],outrand['dec_gal'] = all_RA/degtorad, all_Dec/degtorad
    #del all_RA, allDec
    save_dataframe_to_fits_save(dataframe=outrand, filename=filename, overwrite=True)
    print ('save to', filename)


def create_random_on_sphere( ramin=None, ramax=None, decmin=None, decmax=None, size=None ):
    """
    creating uniform randoms on the upper hemisphere
    all input units are degree
    
    """
    rand1 = np.random.uniform((ramin*np.pi/180)/(2*np.pi), (ramax*np.pi/180)/(2*np.pi), size=size)

    # degree to radian
    decmin_radian = np.pi/180 * decmin 
    decmax_radian = np.pi/180 * decmax 

    phi_min = np.pi/2. - decmin_radian  
    phi_max = np.pi/2. - decmax_radian  

    r2_max = (np.cos(phi_min) + 1.)/2.
    r2_min = (np.cos(phi_max) + 1.)/2.

    rand2 = np.random.uniform(r2_min,r2_max,size=size)
    #randtheta = 2*np.pi*rand1
    randphi = np.arccos(2*rand2-1)
    randdec = np.pi/2 - randphi
    randra = 2*np.pi*rand1

    # DEC
    #decmin = (center[1] - window)/degtorad # Assume RA is 0 -> 360 (see plot)
    #decmax = (center[1] + window)/degtorad

    #randredshift = dist1.generate(randn,verbose=1)
    #randredshift = randredshift[(randredshift > np.min(targets['observed_redshift_gal'])) 
    #                        & (randredshift < np.max(targets['observed_redshift_gal']))]
    #while len(randredshift) < randn:
    #    new = dist1.generate(randn-len(randredshift),verbose=1)
    #    randredshift = np.append(randredshift,new)
    #    randredshift = randredshift[(randredshift > np.min(targets['observed_redshift_gal'])) 
    #                            & (randredshift < np.max(targets['observed_redshift_gal']))]
    #
    #randx = np.cos(randdec)*np.cos(randra)
    #randy = np.cos(randdec)*np.sin(randra)
    #randz = np.sin(randdec)
    return [randra, randdec]






class Estimator_Test:

    def __init__(self, gal=None, gal2=None, rand=None, rand2=None, cosmo=None, ra_col='ra_gal', 
    dec_col='dec_gal', z_col='observed_redshift_gal', smin=1, smax=150, nsbin=50, mode='1d'):

        self.edges = np.linspace(smin,smax,nsbin)
        #Nmu = 20 
        self.mode=mode 
        self.gal = gal 
        self.gal2=gal2
        self.rand = rand 
        self.rand2 = rand2 
        self.cosmo=cosmo 
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.z_col = z_col

        self.ND = gal.size
        self.NR = rand.size
        if self.rand2 is not None:
            self.NR2 = rand2.size 

        self.DD =None
        self.DR = None 
        self.RR = None 
        self.RR_denom = None  

    def run(self):
        from nbodykit.algorithms.pair_counters.mocksurvey import SurveyDataPairCount 

        self.DD = SurveyDataPairCount(self.mode, self.gal, self.edges, cosmo=self.cosmo, second=self.gal, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
            
        self.DR = SurveyDataPairCount(self.mode, self.gal, self.edges, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)

        self.RR = SurveyDataPairCount(self.mode, self.rand, self.edges, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)

        if self.rand2 is not None:
            #self.NR2 = rand2.size
            self.RR_denom = SurveyDataPairCount(self.mode, self.rand2, self.edges, cosmo=self.cosmo, second=self.rand2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
 
        self.r = self.DD.pairs.edges['r']
        #self.mu = self.DD.pairs.edges['mu']
        #dmu = np.abs(self.mu[2] - self.mu[1])
        dr = np.abs(self.r[2] - self.r[1])
        self.r = self.r[:-1] + dr/2.
        #self.mu = self.mu[:-1] + dmu/2.

        """
        self.DD = SurveyDataPairCount('1d', gal, self.edges, cosmo=cosmo, second=gal2, \
            ra=ra_col, dec=dec_col, redshift=z_col, show_progress=True)
            
        self.DR = SurveyDataPairCount('1d', gal, self.edges, cosmo=cosmo, second=rand2, \
            ra=ra_col, dec=dec_col, redshift=z_col, show_progress=True)
        
        if self.gal2 is None: 
            self.RD = self.DR
        elif self.gal2 is not None:
            self.RD = SurveyDataPairCount('1d', gal, self.edges, cosmo=cosmo, second=rand, \
                ra=ra_col, dec=dec_col, redshift=z_col, show_progress=True)

        self.RR = SurveyDataPairCount('1d', rand, edges, cosmo=cosmo, second=rand, \
            ra=ra_col, dec=dec_col, redshift=z_col, show_progress=True)

        if rand2 is not None:
            self.NR2 = rand2.size
            self.RR_denom = SurveyDataPairCount('1d', rand2, edges, cosmo=cosmo, second=rand2, \
            ra=ra_col, dec=dec_col, redshift=z_col, show_progress=True)
        """


    def run_cross(self):
        from nbodykit.algorithms.pair_counters.mocksurvey import SurveyDataPairCount 

        self.D1D2 = SurveyDataPairCount(self.mode, self.gal, self.edges, cosmo=self.cosmo, second=self.gal2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
            
        self.D1R2 = SurveyDataPairCount(self.mode, self.gal, self.edges, cosmo=self.cosmo, second=self.rand2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)

        self.D2R1 = SurveyDataPairCount(self.mode, self.gal2, self.edges, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)

        self.R1R2 = SurveyDataPairCount(self.mode, self.rand, self.edges, cosmo=self.cosmo, second=self.rand2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
 
        self.r = self.D1D2.pairs.edges['r']
        #self.mu = self.DD.pairs.edges['mu']
        #dmu = np.abs(self.mu[2] - self.mu[1])
        dr = np.abs(self.r[2] - self.r[1])
        self.r = self.r[:-1] + dr/2.
        #self.mu = self.mu[:-1] + dmu/2.

        self.ND1 = self.gal.size
        self.NR1 = self.rand.size
        self.ND2 = self.gal2.size
        self.NR2 = self.rand2.size


    def compute_standard_LS_cross(self):
        alpha1 = self.NR1/self.ND2 
        alpha2 = self.NR2/self.ND1

        DDnorm = self.D1D2.pairs['npairs'] * alpha1 * alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.D2R1.pairs['npairs'] * alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        RDnorm = self.D1R2.pairs['npairs'] * alpha2 
        RRnorm = self.R1R2.pairs['npairs'] #/self.RR.pairs.attrs['total_wnpairs']

        xi = (DDnorm - DRnorm - RDnorm + RRnorm)/RRnorm
        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        r = self.D1D2.pairs['r']
        return (r, xi)
        

    def standard_natural_estimator(self):
        alpha1 = self.NR/self.ND 
        alpha2 = self.NR/self.ND

        DDnorm = self.DD.pairs['npairs'] * alpha1 * alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        #DRnorm = self.DR.pairs['npairs'] * alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        #RDnorm = self.DR.pairs['npairs'] * alpha2 
        RRnorm = self.RR.pairs['npairs'] #/self.RR.pairs.attrs['total_wnpairs']

        xi = (DDnorm-RRnorm)/RRnorm
        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        r = self.DD.pairs['r']
        return (r, xi)

    def modified_natural_estimator(self):
        alpha1 = self.NR/self.ND 
        alpha2 = self.NR/self.ND

        DDnorm = self.DD.pairs['npairs'] * alpha1 * alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        #DRnorm = self.DR.pairs['npairs'] * alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        #RDnorm = self.DR.pairs['npairs'] * alpha2 
        RRnorm = self.RR.pairs['npairs'] #/self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs'] 
        xi = (DDnorm-RRnorm)/RRnorm_denominator
        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        r = self.DD.pairs['r']
        return (r, xi)

    def compute_standard_LS(self):
        alpha1 = self.NR/self.ND 
        alpha2 = self.NR/self.ND

        DDnorm = self.DD.pairs['wnpairs'] * alpha1 * alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['wnpairs'] * alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        RDnorm = self.DR.pairs['wnpairs'] * alpha2 
        RRnorm = self.RR.pairs['wnpairs'] #/self.RR.pairs.attrs['total_wnpairs']

        xi = (DDnorm - DRnorm - RDnorm + RRnorm)/RRnorm
        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        r = self.DD.pairs['r']
        return (r, xi)

    def compute_modified_LS(self):
        alpha1 = self.NR/self.ND 
        alpha2 = self.NR/self.ND

        DDnorm = self.DD.pairs['wnpairs'] * alpha1 * alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['wnpairs'] * alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        RDnorm = self.DR.pairs['wnpairs'] * alpha2 
        RRnorm = self.RR.pairs['wnpairs'] #/self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs'] #/self.RR_denom.pairs.attrs['total_wnpairs']

        xi = (DDnorm - DRnorm - RDnorm + RRnorm)/RRnorm_denominator
        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        r = self.DD.pairs['r']
        return (r, xi)

    def _compute_modified_LS(self):
        DDnorm = self.DD.pairs['npairs']/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['npairs']/self.DR.pairs.attrs['total_wnpairs']
        RRnorm = self.RR.pairs['npairs']/self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs']/self.RR_denom.pairs.attrs['total_wnpairs']
        xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm_denominator
        r = self.DD.pairs['r']
        return (r, xi)

    def _sanity_check_compute_modified_LS(self):
        DDnorm = self.DD.pairs['npairs']/ self.gal.size**2 #  self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['npairs']/ (self.gal.size * self.rand.size) #self.DR.pairs.attrs['total_wnpairs']
        RRnorm = self.RR.pairs['npairs']/ self.rand.size**2 # self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs']/ self.rand2.size**2
        xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm_denominator
        r = self.DD.pairs['r']
        return (r, xi)

    def save(self):

        return    



class Twopt_Correlation_Multipoles:

    def __init__(self, gal=None, gal2=None, rand=None, rand2=None, rand_deno=None, md=False, cosmo=None, ra_col='ra_gal', 
    dec_col='dec_gal', z_col='observed_redshift_gal', Nmu=101, mode='2d', smin=1, smax=150, nsbin=50, nthreads=1):

        self.edges = np.linspace(smin,smax,nsbin, dtype=float)
        #Nmu = 20 
        self.mode=mode 
        self.gal = gal 
        self.gal2=gal2
        self.rand = rand 
        self.rand2 = rand2 
        self.rand_deno = rand_deno
        self.cosmo=cosmo 
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.z_col = z_col
        self.Nmu = Nmu 
        self.nthreads = nthreads
        
        # self.ND = gal.size
        # self.NR = rand.size
        # if self.rand2 is not None:
        #     self.NR2 = rand2.size 

        # computing the total gal and random size by summing over samples across multiple cpus 
        self.ND = np.sum( self.gal.comm.allgather( self.gal.size) ) 
        self.NR = np.sum( self.rand.comm.allgather( self.rand.size) ) 
        self.alpha1 = self.NR/self.ND
        self.alpha2 = self.NR/self.ND
        if self.gal2 is not None:
            self.ND2 = np.sum( self.gal2.comm.allgather( self.gal2.size ) )
        if self.rand2 is not None:
            self.NR2 = np.sum( self.rand2.comm.allgather( self.rand2.size ) )

        self.DD =None
        self.DR = None 
        self.RR = None 
        self.RR_denom = None  
        
        self.D1D2 = None
        self.D1R2 = None
        self.D2R1 = None
        self.R1R2 = None

    def run(self):
        from nbodykit.algorithms.pair_counters.mocksurvey import SurveyDataPairCount 

        #if self.gal2 is None: 
        #    self.gal2 = self.gal
        #    self.rand2 = self.rand
        #else: 
        if self.gal2 is not None:
            logging.info('Two galaxy samples are given. Computing cross-correlation...')
            self.run_cross()
            return 0

        #logging.info('Counting DD pairs')
        self.DD = SurveyDataPairCount(self.mode, self.gal, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.gal, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
        #logging.info('Counting DR pairs')    
        self.DR = SurveyDataPairCount(self.mode, self.gal, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
        #logging.info('Counting RR pairs')
        #self.RD = self.DR
        self.RR = SurveyDataPairCount(self.mode, self.rand, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)

        if self.rand_deno is not None:
            #self.NR2 = rand2.size
            logging.info('Counting RR pairs for denominator')
            self.RR_denom = SurveyDataPairCount(self.mode, self.rand_deno, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand_deno, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
 
        self.r = self.DD.pairs.edges['r']
        self.mu = self.DD.pairs.edges['mu']
        dmu = np.abs(self.mu[2] - self.mu[1])
        dr = np.abs(self.r[2] - self.r[1])
        self.r = self.r[:-1] + dr/2.
        self.mu = self.mu[:-1] + dmu/2.

        #self.ND = np.sum( self.gal.comm.allgather( self.gal.compute(self.gal[self.weight_col].sum()) ) )
        #self.NR = np.sum( self.rand.comm.allgather( self.rand.compute(self.rand[self.weight_col].sum()) ) )
        #self.alpha = self.ND/self.NR
 

    def run_cross(self):
        from nbodykit.algorithms.pair_counters.mocksurvey import SurveyDataPairCount 

        #logging.info('Counting DD pairs')
        self.D1D2 = SurveyDataPairCount(self.mode, self.gal, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.gal2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)  
        self.D1R2 = SurveyDataPairCount(self.mode, self.gal, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
        self.D2R1 = SurveyDataPairCount(self.mode, self.gal2, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
        self.R1R2 = SurveyDataPairCount(self.mode, self.rand, self.edges, Nmu=self.Nmu, cosmo=self.cosmo, second=self.rand2, \
            ra=self.ra_col, dec=self.dec_col, redshift=self.z_col, show_progress=True)
 
        self.r = self.D1D2.pairs.edges['r']
        self.mu = self.D1D2.pairs.edges['mu']
        dmu = np.abs(self.mu[2] - self.mu[1])
        dr = np.abs(self.r[2] - self.r[1])
        self.r = self.r[:-1] + dr/2.
        self.mu = self.mu[:-1] + dmu/2.

        # self.ND1 = self.gal.size
        # self.NR1 = self.rand.size
        # self.ND2 = self.gal2.size
        # self.NR2 = self.rand2.size


    def save(self, outdir=None):
        if self.DD is not None: self.DD.save(outdir+'DD.json')
        if self.DR is not None: self.DR.save(outdir+'DR.json')
        if self.RR is not None: self.RR.save(outdir+'RR.json')
        if self.RR_denom is not None: self.RR_denom.save(outdir+'RR_denom.json')
        if self.D1D2 is not None: self.D1D2.save(outdir+'D1D2.json')
        if self.D1R2 is not None: self.D1D2.save(outdir+'D1R2.json')
        if self.D2R1 is not None: self.D1D2.save(outdir+'D2R1.json')
        if self.R1R2 is not None: self.D1D2.save(outdir+'R1R2.json')


    def compute_standard_LS(self):
        # alpha1 = self.NR/self.ND 
        # alpha2 = self.NR/self.ND
        logging.info('Computing standard LS estimator')
        DDnorm = self.DD.pairs['wnpairs'] * self.alpha1 * self.alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['wnpairs'] * self.alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        #RDnorm = self.DR.pairs['wnpairs'] * self.alpha2 
        RRnorm = self.RR.pairs['wnpairs'] #/self.RR.pairs.attrs['total_wnpairs']
        #logging.info('Computing standard LS estimator')
        # DDnorm = self.DD.pairs['wnpairs'] /self.DD.pairs.attrs['total_wnpairs']
        # DRnorm = self.DR.pairs['wnpairs'] /self.DR.pairs.attrs['total_wnpairs']
        # RDnorm = self.DR.pairs['wnpairs'] /self.DR.pairs.attrs['total_wnpairs']
        # RRnorm = self.RR.pairs['wnpairs'] /self.RR.pairs.attrs['total_wnpairs']

        #print ( 'DDnorm', DDnorm[:,0] )
        #print ( 'DDweight', self.DD.pairs['wnpairs'][:,0] )
        #print ( 'DDnpairs', self.DD.pairs['npairs'][:,0] )

        xi_smu = (DDnorm - 2*DRnorm + RRnorm)/RRnorm
        #print ('in compute_standard_LS: NR, ND=', self.NR, self.ND)
        #print ('in compute_standard_LS: xi=', xi)
        return xi_smu

    def compute_modified_LS(self):
        # alpha1 = self.NR/self.ND 
        # alpha2 = self.NR/self.ND
        logging.info('Computing modified LS estimator')
        DDnorm = self.DD.pairs['wnpairs'] * self.alpha1 * self.alpha2 #/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['wnpairs'] * self.alpha1 #/self.DR.pairs.attrs['total_wnpairs']
        #RDnorm = self.DR.pairs['wnpairs'] * self.alpha2 
        RRnorm = self.RR.pairs['wnpairs'] #/self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['wnpairs'] #/self.RR_denom.pairs.attrs['total_wnpairs']
        #logging.info('Computing modified LS estimator')
        # DDnorm = self.DD.pairs['wnpairs'] /np.sqrt(self.DD.pairs.attrs['total_wnpairs'])
        # DRnorm = self.DR.pairs['wnpairs'] /np.sqrt(self.DR.pairs.attrs['total_wnpairs'])
        # RDnorm = self.DR.pairs['wnpairs'] /np.sqrt(self.DR.pairs.attrs['total_wnpairs'])
        # RRnorm = self.RR.pairs['wnpairs'] /np.sqrt(self.RR.pairs.attrs['total_wnpairs'])
        # RRnorm_denominator = self.RR_denom.pairs['wnpairs'] /np.sqrt(self.RR_denom.pairs.attrs['total_wnpairs'])

        xi_smu = (DDnorm - 2* DRnorm + RRnorm)/RRnorm_denominator

        #xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm
        #r = self.DD.pairs['r']
        return xi_smu

    def compute_standard_LS_cross(self):

        # For now, I am using total_wnpairs as I have no idea how to compute alpha parameter
        # when multiple samples are involved. Will correct it later 
        logging.info('Computing standard LS estimator for cross-correlation')
        D1D2norm = self.D1D2.pairs['wnpairs'] /self.D1D2.pairs.attrs['total_wnpairs']
        D1R2norm = self.D1R2.pairs['wnpairs'] /self.D1R2.pairs.attrs['total_wnpairs']
        D2R1norm = self.D2R1.pairs['wnpairs'] /self.D2R1.pairs.attrs['total_wnpairs']
        R1R2norm = self.R1R2.pairs['wnpairs'] /self.R1R2.pairs.attrs['total_wnpairs']

        #xi = (DDnorm - DRnorm - RDnorm + RRnorm)/RRnorm
        xi_smu = (D1D2norm - D1R2norm - D2R1norm + R1R2norm)/R1R2norm
        #print ('in compute_standard_LS: NR, ND=', self.NR, self.ND)
        #print ('in compute_standard_LS: xi=', xi)
        
        return xi_smu

    def standard_multipoles(self):

        xi_smu = self.compute_standard_LS()

        self.legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in self.mu]).ravel() 
        self.legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in self.mu]).ravel() 

        self.xi_monopole = np.sum(xi_smu * self.legendre0.T, axis = 1)/self.mu.size
        self.xi_quadrupole = 5.0*np.sum(xi_smu * self.legendre2.T, axis = 1)/self.mu.size

    def modified_multipoles(self):

        xi_smu = self.compute_modified_LS()

        self.legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in self.mu]).ravel() 
        self.legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in self.mu]).ravel() 

        self.xi_modified_monopole = np.sum(xi_smu * self.legendre0.T, axis = 1)/self.mu.size
        self.xi_modified_quadrupole = 5.0*np.sum(xi_smu * self.legendre2.T, axis = 1)/self.mu.size

    def standard_multipoles_cross(self):

        xi_smu = self.compute_standard_LS_cross()

        self.legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in self.mu]).ravel() 
        self.legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in self.mu]).ravel() 

        self.xi_monopole_cross = np.sum(xi_smu * self.legendre0.T, axis = 1)/self.mu.size
        self.xi_quadrupole_cross = 5.0*np.sum(xi_smu * self.legendre2.T, axis = 1)/self.mu.size

    def _compute_modified_LS(self):
        DDnorm = self.DD.pairs['npairs']/self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['npairs']/self.DR.pairs.attrs['total_wnpairs']
        RRnorm = self.RR.pairs['npairs']/self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs']/self.RR_denom.pairs.attrs['total_wnpairs']
        xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm_denominator
        r = self.DD.pairs['r']
        return (r, xi)

    def _sanity_check_compute_modified_LS(self):
        DDnorm = self.DD.pairs['npairs']/ self.gal.size**2 #  self.DD.pairs.attrs['total_wnpairs']
        DRnorm = self.DR.pairs['npairs']/ (self.gal.size * self.rand.size) #self.DR.pairs.attrs['total_wnpairs']
        RRnorm = self.RR.pairs['npairs']/ self.rand.size**2 # self.RR.pairs.attrs['total_wnpairs']
        RRnorm_denominator = self.RR_denom.pairs['npairs']/ self.rand2.size**2
        xi = (DDnorm - 2* DRnorm + RRnorm)/RRnorm_denominator
        r = self.DD.pairs['r']
        return (r, xi)
