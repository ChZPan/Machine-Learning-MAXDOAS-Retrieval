import os
import netCDF4 as nc
import pandas as pd
import glob
import metpy.calc
from metpy.units import units
import numpy as np
from datetime import datetime, timedelta
import shutil
from sklearn.preprocessing import MinMaxScaler

CMD = os.getcwd()
DATAPATH = CMD + "/data/"   # raw retrieval + meteorological data
INPATH = CMD + "/input/"    # input to ML model
OUTPATH = CMD + "/output/"  # output from ML model
AEROPATH = CMD + "/data/aerosol/"
NO2PATH = CMD + "/data/no2/"
METPATH = CMD + "/data/met/"


## Grouping data by quality
GROUPING = False
if GROUPING:
    filters = {'dof_aero': [1.0, 1.4],
               'chisq_aero': [25.0, 55.0],
               'dof_no2': [1.8, 2.2],
               'chisq_no2': [12.0, 22.0]}

    TARGET = 'both'
    xpath = INPATH + 'full_raw/X/'
    ypath = INPATH + 'full_raw/Y/'
    xlist = glob.glob(xpath + '*npz')
    ylist = glob.glob(ypath + '*npz')


    if TARGET == 'no2':
        xhigh = INPATH + 'high_no2/X/'
        yhigh = INPATH + 'high_no2/Y/'
        os.makedirs(xhigh, exist_ok=True)
        os.makedirs(yhigh, exist_ok=True)

        for xfile, yfile in zip(xlist, ylist):

            y = np.load(yfile)
            dof = y['dof_no2']
            chisq = y['chisq_no2']

            if ((dof >= filters['dof_no2'][1]) & (chisq <= filters['chisq_no2'][1])) | \
            ((dof >= filters['dof_no2'][0]) & (chisq <= filters['chisq_no2'][0])):
                shutil.copy2(xfile, xhigh)
                shutil.copy2(yfile, yhigh)

    elif TARGET == 'aero':
        xhigh = INPATH + 'high_aero/X/'
        yhigh = INPATH + 'high_aero/Y/'
        os.makedirs(xhigh, exist_ok=True)
        os.makedirs(yhigh, exist_ok=True)

        for xfile, yfile in zip(xlist, ylist):

            y = np.load(yfile)
            dof = y['dof_aero']
            chisq = y['chisq_aero']

            if ((dof >= filters['dof_aero'][1]) & (chisq <= filters['chisq_aero'][1])) | \
            ((dof >= filters['dof_aero'][0]) & (chisq <= filters['chisq_aero'][0])):
                shutil.copy2(xfile, xhigh)
                shutil.copy2(yfile, yhigh)

    elif TARGET == 'both':
        xboth = INPATH + 'highboth_no2/X/'
        yboth = INPATH + 'highboth_no2/Y/'
        os.makedirs(xboth, exist_ok=True)
        os.makedirs(yboth, exist_ok=True)

        no2_xlist = glob.glob(INPATH + 'high_no2/X/*')
        no2_ylist = glob.glob(INPATH + 'high_no2/Y/*')
        for fx, fy in zip(no2_xlist, no2_ylist):
            fname = os.path.split(fx)[1]
            path = INPATH + "high_aero/X/"
            file = os.path.join(path, fname)
            if os.path.exists(file):
                shutil.copy2(fx, xboth)
                shutil.copy2(fy, yboth)

## Normalization
## normalize all input data into the range of [-1, 1]
NORMALIZATION = False
if NORMALIZATION:
    xflist = glob.glob(INPATH + "normalized/highboth_no2/X/*npz")
    yflist = glob.glob(INPATH + "normalized/highboth_no2/Y/*npz")

    cols_1 = ['dt', 'd2m', 't2m', 'skt', 'sp', 'sza', 'raa', 'o4']
    cols_2 = ['dt', 'tprof', 'qprof', 'uprof', 'vprof', 'aero', 'no2']
    cols_3 = ['dt', 'chisq_aero', 'dof_aero', 'chisq_no2', 'dof_no2']
    cols = [cols_1, cols_2, cols_3]
    df1 = pd.DataFrame(columns=cols_1)
    df2 = pd.DataFrame(columns=cols_2)
    df3 = pd.DataFrame(columns=cols_3)

    for xf, yf in zip(xflist, yflist):
        dt = os.path.split(xf)[1][2:-4]

        dt_arr = np.repeat(np.array(dt), 9)
        d2m = np.load(xf)['d2m']
        t2m = np.load(xf)['t2m']
        skt = np.load(xf)['skt']
        sp = np.load(xf)['sp']
        sza = np.load(xf)['sza']
        raa = np.load(xf)['raa']
        o4 = np.load(xf)['o4']

        df1_temp = pd.DataFrame({'dt': dt, 'd2m': d2m, 't2m': t2m,
                                'skt': skt, 'sp': sp, 'sza': sza,
                                'raa': raa, 'o4': o4})
        df1 = pd.concat([df1, df1_temp], ignore_index=True)

        dt_arr = np.repeat(np.array(dt), 21)
        tprof = np.load(xf)['tprof']
        qprof = np.load(xf)['qprof']
        uprof = np.load(xf)['uprof']
        vprof = np.load(xf)['vprof']
        aero = np.load(xf)['aero']
        no2 = np.load(yf)['no2']

        df2_temp = pd.DataFrame({'dt': dt, 'tprof': tprof, 'qprof': qprof,
                                 'uprof': uprof, 'vprof': vprof,
                                 'aero': aero, 'no2': no2})
        df2 = pd.concat([df2, df2_temp], ignore_index=True)

        chisq_aero = np.load(yf)['chisq_aero']
        dof_aero = np.load(yf)['dof_aero']
        chisq_no2 = np.load(yf)['chisq_no2']
        dof_no2 = np.load(yf)['dof_no2']

        df3_temp = pd.DataFrame({'dt': dt, 'chisq_aero': [chisq_aero], 'dof_aero': [dof_aero],
                                 'chisq_no2': [chisq_no2], 'dof_no2': [dof_no2]})
        df3 = pd.concat([df3, df3_temp], ignore_index=True)

    dfs = [df1, df2, df3]

    for i, df in zip(range(len(dfs)), dfs):
        df.to_csv('df%s.csv' % (i+1), index=False)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_scl = df.copy()
        df_scl[cols[i][1:]] = scaler.fit_transform(df[cols[i][1:]])
        df_scl.to_csv('df%s_scl.csv' % (i+1), index=False)
    df1_scl = pd.read_csv('df1_scl.csv')
    df2_scl = pd.read_csv('df2_scl.csv')
    df3_scl = pd.read_csv('df3_scl.csv')

    for xf, yf in zip(xflist, yflist):
        dt = os.path.split(xf)[1][2:-4]
        xdict = dict(np.load(xf))
        xdict['d2m'] = df1_scl.loc[df1_scl['dt'] == dt]['d2m']
        xdict['t2m'] = df1_scl.loc[df1_scl['dt'] == dt]['t2m']
        xdict['skt'] = df1_scl.loc[df1_scl['dt'] == dt]['skt']
        xdict['sp'] = df1_scl.loc[df1_scl['dt'] == dt]['sp']
        xdict['sza'] = df1_scl.loc[df1_scl['dt'] == dt]['sza']
        xdict['raa'] = df1_scl.loc[df1_scl['dt'] == dt]['raa']
        xdict['o4'] = df1_scl.loc[df1_scl['dt'] == dt]['o4']
        xdict['tprof'] = df2_scl.loc[df2_scl['dt'] == dt]['tprof']
        xdict['qprof'] = df2_scl.loc[df2_scl['dt'] == dt]['qprof']
        xdict['uprof'] = df2_scl.loc[df2_scl['dt'] == dt]['uprof']
        xdict['vprof'] = df2_scl.loc[df2_scl['dt'] == dt]['vprof']
        xdict['aero'] = df2_scl.loc[df2_scl['dt'] == dt]['aero']
        np.savez(xf, **xdict)

        ydict = dict(np.load(yf))
        ydict['aero'] = df2_scl.loc[df2_scl['dt'] == dt]['aero']
        ydict['no2'] = df2_scl.loc[df2_scl['dt'] == dt]['no2']
        ydict['chisq_aero'] = df3_scl.loc[df3_scl['dt'] == dt]['chisq_aero']
        ydict['dof_aero'] = df3_scl.loc[df3_scl['dt'] == dt]['dof_aero']
        ydict['chisq_no2'] = df3_scl.loc[df3_scl['dt'] == dt]['chisq_no2']
        ydict['dof_no2'] = df3_scl.loc[df3_scl['dt'] == dt]['dof_no2']
        np.savez(yf, **ydict)


SCALING = True
if SCALING:
    xflist = glob.glob(INPATH + "scaled/highboth_no2/X/*npz")
    yflist = glob.glob(INPATH + "scaled/highboth_no2/Y/*npz")
    for xf, yf in zip(xflist, yflist):
        dt = os.path.split(xf)[1][2:-4]
        xdict = dict(np.load(xf))
        xdict['d2m'] /= 100
        xdict['t2m'] /= 100
        xdict['skt'] /= 100
        xdict['sp'] /= 1e5
        xdict['sza'] /= 100
        xdict['raa'] /= 100
        xdict['o4'] *= 100
        xdict['tprof'] /= 100
        xdict['qprof'] *= 1e3
        xdict['uprof'] /= 10
        xdict['vprof'] /= 10
        np.savez(xf, **xdict)

        ydict = dict(np.load(yf))
        ydict['no2'] /= 1e11
        np.savez(yf, **ydict)





