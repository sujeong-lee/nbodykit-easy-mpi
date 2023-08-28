import nbodykit as nb # Installed in my pypelid environment
from nbodykit.lab import *
from nbodykit.algorithms.pair_counters.mocksurvey import SurveyDataPairCount 
from nbodykit import CurrentMPIComm
import sys, os, time, yaml, argparse, warnings, logging
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
#logging.warning('Watch out!')  # will print a message to the console
#logging.info('I told you so') 
#logging.basicConfig(filename='example.log', level=logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

sys.path.append('../')
from nbodykit_lib import Twopt_Correlation_Multipoles

def cut_euclid_zrange(cat, zrange=[0.8, 1.8], ztag='observed_redshift_gal'):
    mask = (cat['observed_redshift_gal'] > zrange[0]) & (cat['observed_redshift_gal'] < zrange[1])
    return cat[mask]

def test():
    print ('Hello, World')

def main(comm, config):

    # Choose cosmology for the conversion from z to distance
    t1 = time.time()
    #if comm.rank==0: 
    gal_filename = config['gal']
    rand_filename = config['random']
    #Nmu  = config['Nmu']
    #pimax= config['pimax']
    #smin = config['smin']
    #smax = config['smax']
    #nsbin= config['nsbin']
    zrange=config['redshift']
    z_col = config['z_col']

    if 'gal2' not in list(config.keys() ): config['gal2']=None
    if 'rand2' not in list(config.keys() ): config['rand2']=None
    if 'rand_deno' not in list(config.keys() ): config['rand_deno']=None

    cosmo = nb.lab.cosmology.Cosmology(Omega0_cdm=0.26185743, Omega0_b=0.04814257, h=0.676)

    if comm.rank == 0:
        logging.info('name={}'.format(config['name']))
        logging.info('gal_fname={}'.format(gal_filename))
        logging.info('rand_fname={}'.format(rand_filename))

        # check the directory to save output exists
        #outdir = os.path.join(*config['savedir'].split('/'))
        #if os.path.exists(outdir): pass 
        #else: 
        #os.makedirs(outdir, exist_ok=True)
        #logging.info('creating output directory = {}'.format(outdir))
        #exit(1)
        logging.info('cosmology={}'.format(cosmo))

    # catalog
    ogalcat = nb.lab.FITSCatalog(gal_filename)
    orandcat = nb.lab.FITSCatalog(rand_filename)
    
    # Mask catastrophic redshifts
    # cutting from 0.8 to 1.8 on 'observed_redshift_gal
    ogalcat = cut_euclid_zrange(ogalcat, zrange=zrange)
    orandcat = cut_euclid_zrange(orandcat, zrange=zrange)

    # # gal2 and random2 ---------------------------------------
    # if config['gal2'] is None: ogalcat2=ogalcat 
    # else: 
    #     ogalcat2 = nb.lab.FITSCatalog(str(config['gal2']))
    #     ogalcat2 = cut_euclid_zrange(ogalcat2, zrange=zrange)
    #
    # if config['random2'] is None: orandcat2=orandcat 
    # else: 
    #     orandcat2 = nb.lab.FITSCatalog(str(config['random2']) )
    #     orandcat2 = cut_euclid_zrange(orandcat2, zrange=zrange)
    # # gal2 and random2 ---------------------------------------

    # cross
    ogalcat2=None; orandcat2=None; orandcat_deno = None
    if (config['gal2'] is not None): 
        ogalcat2  = nb.lab.FITSCatalog(config['gal2'])
        orandcat2 = nb.lab.FITSCatalog(config['random2'])
        ogalcat2  = cut_euclid_zrange(ogalcat2, zrange=zrange)
        orandcat2 = cut_euclid_zrange(orandcat2, zrange=zrange)
        logging.info('--rank={} : {:7} gals / {:7} gals2 / {:7} randoms / {:7} randoms2'.format(comm.rank, ogalcat.size, ogalcat2.size, orandcat.size,  orandcat2.size))

    elif config['random_denominator'] is not None: 
        orandcat_deno = nb.lab.FITSCatalog(str(config['random_denominator']) )
        orandcat_deno = cut_euclid_zrange(orandcat_deno, zrange=zrange)
        logging.info('--rank={} : {:7} gals / {:7} randoms / {:7} randoms_denominator'.format(comm.rank, ogalcat.size, orandcat.size,  orandcat_deno.size))

    else: 
        logging.info('--rank={} : {:7} gals / {:7} randoms'.format(comm.rank, ogalcat.size, orandcat.size))

    # Pair counting
    ET_Pole = Twopt_Correlation_Multipoles(gal=ogalcat, rand=orandcat, gal2=ogalcat2, rand2=orandcat2, rand_deno=orandcat_deno, 
    cosmo=cosmo, Nmu=100, mode='2d', z_col=z_col)
    ET_Pole.run()
    ET_Pole.save(outdir=outdir)

    if comm.rank == 0:
        # convert pair counting to multipoles        
        if ET_Pole.gal2 is not None: 
            ET_Pole.standard_multipoles_cross()
            # saving the result
            DAT2 = np.column_stack(( ET_Pole.r,  ET_Pole.xi_monopole_cross,  ET_Pole.xi_quadrupole_cross ))
            header = 'r (Mpc/h)     xi0     xi2'
            np.savetxt(outdir+'xi_standard_cross.dat', DAT2, fmt='%.8e', header=header, delimiter='    ')
            logging.info('measurement done; saving result to {}'.format(outdir+'xi_standard_cross.dat'))

        else: 
            ET_Pole.standard_multipoles()
            # saving the result
            DAT = np.column_stack(( ET_Pole.r,  ET_Pole.xi_monopole,  ET_Pole.xi_quadrupole ))
            header = 'r (Mpc/h)     xi0     xi2'
            np.savetxt(outdir+'xi_standard.dat', DAT, fmt='%.8e', header=header, delimiter='    ')
            logging.info('measurement done; saving result to {}'.format(outdir+'xi_standard.dat'))

            if ET_Pole.rand_deno is not None: 
                ET_Pole.modified_multipoles()
                # saving the result
                DAT2 = np.column_stack(( ET_Pole.r,  ET_Pole.xi_modified_monopole,  ET_Pole.xi_modified_quadrupole ))
                header = 'r (Mpc/h)     xi0     xi2'
                np.savetxt(outdir+'xi_modified.dat', DAT2, fmt='%.8e', header=header, delimiter='    ')
                logging.info('measurement done; saving result to {}'.format(outdir+'xi_modified.dat'))

        logging.info('computing time = {:0.1f} s'.format(time.time()-t1))

if __name__ == '__main__':

    """
    desc = "an nbodykit example script using the TaskManager class"
    parser = argparse.ArgumentParser(description=desc)

    #h = 'the number of cpus per task'
    #parser.add_argument('cpus_per_task', type=int, help=h)

    parser.add_argument('-l', '--log', type=str, help= 'log filename')
    parser.add_argument('-o', '--output', default='./', type=str, help= 'output directory to save. Default is ./')


    args = parser.parse_args()

    #logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    if args.log == None: 
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    if args.log is not None:
        logging.basicConfig(filename=args.log, level=logging.INFO)

    comm = CurrentMPIComm.get()
    main(comm, args)

    """
    
    desc = "an nbodykit example script using the TaskManager class"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('configfile', type=str, help= 'yaml configuration filename')
    args = parser.parse_args()
    comm = CurrentMPIComm.get()

    with open(args.configfile, 'r') as file:
        #configs = file('config.yaml', 'r') 
        config = yaml.load(file, Loader=yaml.Loader)

    # check whether output dir exists
    outdir = str(config['savedir']) 
    #if comm.rank == 0:
        #if os.path.exists(outdir): pass
    if not os.path.exists(outdir): 
        os.makedirs(outdir, exist_ok=True)
        logging.info('creating output dir: {}'.format(outdir))
    # check the directory to save output exists
    # outputdir = os.path.join(*config['output_fname'].split('/')[:-1])
    # if os.path.exists(outputdir): pass 
    # else: os.makedirs(outputdir)

    # if comm.rank == 0:
    #     if os.path.exists(outdir): pass
    #     else:
    #         try: 
    #             os.mkdir(outdir)
    #             logging.info('creating output dir: {}'.format(outdir))
    #             print ('creating output dir: {}'.format(outdir))
    #         except (FileExistsError): 
    #             print ('Something went wrong!')
    #             exit(1)
          
    import logging
    
    logger = logging.getLogger()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    # # stream logger
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(logging.INFO)
    # logger.addHandler(stream_handler)

    # logfile loader
    if config['logfile'] is not None: 
        file_handler = logging.FileHandler(outdir+config['logfile'])
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

    #logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    #if config['logfile'] is None: 
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    #else:
    #    logging.basicConfig(filename=str(outdir+config['logfile']), level=logging.INFO)

    #print (config['gal'])
    main(comm, config)