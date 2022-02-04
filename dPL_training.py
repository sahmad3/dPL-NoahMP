import sys
from hydroDL import master, utils
from hydroDL.data import dbCsv
import hydroDL.data as hd
from hydroDL.master import default, wrapMaster, runTrain, test
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat

import os
from numpy import newaxis
import numpy as np
import statistics
import torch
import matplotlib.pyplot as plt
import pandas as pd
import csv

# GPU setting
traingpuid = 0
torch.cuda.set_device(traingpuid)
 
## database
cDir = os.getcwd()
rootDB = os.path.join(cDir, 'data')
## save path
saveResultsPath = cDir + '/dPL_results/'
if os.path.exists(saveResultsPath) is False:
    os.mkdir(saveResultsPath)

### get arguments
simulation= 'sim1'
epoch_surrogate=50
epoch_dPL=500
print('dPL training ......... \n sim: {}, epoch_surrogate:{}, epoch_dPL'.format(simulation,epoch_surrogate,epoch_dPL))
######
    
subsetLst=['TEST_DOM']
## for dPL
tRangeLst=[[20150402, 20180401]]  ## training period
tRangeLstTest=[[20180402, 20210401]]  ## testing period
## for near-real-time forecast method
tRangeLst2=[[20150401, 20180331]]  ## training period
tRangeLst2Test=[[20180401, 20210331]]  ## testing period

## hidden size of dPL
hiddenSizeLst = [256]
hsLst = [256]

## dPL target
targetVar=['SMAP_PM']

# training year
trainYear=int(str(tRangeLst[0][1])[0:4])-int(str(tRangeLst[0][0])[0:4])


df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLst[0]
)
df_test = dbCsv.DataframeCsv(
    rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLstTest[0]
)

## =========== read Surrogate model
filename = cDir +  '/surrogate_models/smsurrogate_' + simulation + '_Ep'+str(epoch_surrogate)+'.pt'  # 

## =========== save trained dPL model
modelName = 'dPL_soilm_smsurrogate_' + simulation
 

# ===== Noah-MP varForcing, dynamic inputs =====

varForcing_Noah = [
'lis_Rainf_f_tavg',
'lis_LWdown_f_tavg',
'lis_SWdown_f_tavg',
'lis_Tair_f_tavg',
'lis_Qair_f_tavg',
'lis_Wind_f_tavg',
'lis_Qle_tavg',
'lis_Qh_tavg']


#################################################
##  dicts to hold all calibration parameters and bounds --- NO CHANGE TO PARAMETERS HERE
allConst = ['smcmax', 'psisat', 'dksat', 'dwsat', 'bexp', 'quartz', 'rs', 'rgl', 'hs', 'z0min', 'z0max', 'laimin', 'laimax', 'snup', 'cfactr', 'cmcmax', 'rsmax', 'topt', 'sbeta', 'refdk', 'fxexp', 'refkdt', 'czil', 'csoil', 'frzk', 'sh2o1', 'sh2o2', 'sh2o3', 'sh2o4', 'smc1', 'smc2', 'smc3', 'smc4']
lbConst =  [0.3, 0.01, 5e-07, 5.71e-06, 3.0, 0.1, 40, 30, 36.35, 0.01, 0.01, 0.05, 0.05, 0.02, 0.1, 0.0001, 2000, 293, -4, 5e-07, 0.2, 0.1, 0.05, 1260000.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
ubConst =  [0.55, 0.7, 3e-05, 2.33e-05, 9.0, 0.9, 1000, 150, 55, 0.99, 0.99, 6.0, 6.0, 0.08, 2.0, 0.002, 10000, 303, -1, 3e-05, 4.0, 10.0, 0.8, 3500000.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

zip_iterator = zip(allConst,lbConst)
dict_lbConst = dict(zip_iterator)

zip_iterator = zip(allConst,ubConst)
dict_ubConst = dict(zip_iterator)
#################################################


# ===== parameters of Noah-MP =====

varConst_Noah = ['smcmax','psisat','dksat','dwsat','bexp','quartz',  'rgl', 'z0min','laimin',          
'snup','cfactr','cmcmax',         'topt', 'sbeta','refdk','fxexp', 'refkdt', 'czil',       'frzk',
'sh2o1','sh2o2','sh2o3','sh2o4', 'smc1','smc2','smc3','smc4']     


ub = np.array([dict_ubConst[k] for k in varConst_Noah] + [10, 10])    ##last two for a,b for sm correction y = ax+b
lb = np.array([dict_lbConst[k] for k in varConst_Noah] + [-10, -10])
 

# ===== static attributes for gap-filling SMAP=====
varRaw = [
    'WISE_SILT', 'WISE_SAND', 'WISE_CLAY', 'WISE_TAWC', 'WISE_BULK', 'lis_LAI_inst',
    'flag_SMAP_roughness', 'flag_SMAP_waterbody', 'flag_SMAP_albedo', 'lis_Landcover_inst', 'flag_SMAP_staWater',
    'flag_SMAP_vegDense', 'flag_SMAP_urban', 'flag_SMAP_mount', 'flag_SMAP_ice', 'flag_SMAP_coast'
]


## read database
forcing = df.getDataTs(varForcing_Noah, doNorm=True, rmNan=True)
parameters = df.getDataConst(varConst_Noah, doNorm=True, rmNan=True)  
rawData = df.getDataConst(varRaw, doNorm=True, rmNan=True)
target = df.getDataTs([targetVar[0]], doNorm=True, rmNan=False)
target_GF = df.getDataTs([targetVar[0]], doNorm=True, rmNan=False)

forcingTest = df_test.getDataTs(varForcing_Noah, doNorm=True, rmNan=True)
parametersTest = df_test.getDataConst(varConst_Noah, doNorm=True, rmNan=True)   
rawDataTest = df_test.getDataConst(varRaw, doNorm=True, rmNan=True)
targetTest = df_test.getDataTs([targetVar[0]], doNorm=True, rmNan=False)
targetTest_GF = df_test.getDataTs([targetVar[0]], doNorm=True, rmNan=False)

########################
# near-real-time forecast method (DA), preparing for the dPL
df2 = dbCsv.DataframeCsv(
    rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLst2[0]
)
df2_test = dbCsv.DataframeCsv(
    rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLst2Test[0]
)

# ===== input of DA, Noah grid
varForcing2 = [
    'lis_Rainf_f_tavg','lis_LWdown_f_tavg','lis_SWdown_f_tavg','lis_Tair_f_tavg', 'lis_Qair_f_tavg',
    'lis_NWind_f_tavg','lis_EWind_f_tavg','lis_PotEvap_tavg','lis_Psurf_f_tavg'
]

varConst2 = [
    'WISE_SILT', 'WISE_SAND', 'WISE_CLAY', 'WISE_TAWC', 'WISE_BULK', 'lis_LAI_inst',
    'flag_SMAP_roughness', 'flag_SMAP_waterbody', 'flag_SMAP_albedo', 'lis_Landcover_inst', 'flag_SMAP_staWater',
    'flag_SMAP_vegDense', 'flag_SMAP_urban', 'flag_SMAP_mount', 'flag_SMAP_ice', 'flag_SMAP_coast'
]
forcing2 = df.getDataTs(varForcing2, doNorm=True, rmNan=True)
const2 = df.getDataConst(varConst2, doNorm=True, rmNan=True)
obs_pre = df2.getDataTs([targetVar[0]], doNorm=True, rmNan=False)

## read model for near-real-time forecast (DA)
forecastModelFile = cDir + '/model_path/DA_model/DA_model_smap.pt'
forecastModel = torch.load(forecastModelFile, map_location='cuda:3')
train.testModel(forecastModel, (forcing2, obs_pre), const2,
                filePathLst=['outDA_'+subsetLst[0]+'_tr'+str(trainYear)+'_'+str(hsLst[0])+'_target_woGF'], batchSize=100)

with open(cDir + '/outDA_'+subsetLst[0]+'_tr'+str(trainYear)+'_'+str(hsLst[0])+'_target_woGF') as f:
    reader = csv.reader(f, delimiter=',')
    out0_temp = list(reader)
    outRes_temp = np.array(out0_temp).astype(float)
outRes = outRes_temp[:, :, newaxis]

target_GF[target_GF != target_GF] = outRes[target_GF != target_GF]
modelTarget = target

forcing2Test = df_test.getDataTs(varForcing2, doNorm=True, rmNan=True)
const2Test = df_test.getDataConst(varConst2, doNorm=True, rmNan=True)
obs_preTest = df2_test.getDataTs([targetVar[0]], doNorm=True, rmNan=False)
train.testModel(forecastModel, (forcing2Test, obs_preTest), const2Test,
                filePathLst=['outDATest_'+subsetLst[0]+'_tr'+str(trainYear)+'_'+str(hsLst[0])
                             +'_target_woGF'], batchSize=100)

with open(cDir + '/outDATest_'+subsetLst[0]+'_tr'+str(trainYear)+'_'+str(hsLst[0])
          +'_target_woGF') as f:
    reader = csv.reader(f, delimiter=',')
    out0_temp_test = list(reader)
    outRes_temp_test = np.array(out0_temp_test).astype(float)
outResTest = outRes_temp_test[:, :, newaxis]

targetTest_GF[targetTest_GF != targetTest_GF] = outResTest[targetTest_GF != targetTest_GF]
modelTargetTest = targetTest

## =========== read Surrogate model
# filename = cDir +  '/output/TEST_DOM/model_surrogate_soilonly_Ep20.pt'
model_loaded_PF_SMAP = torch.load(filename)
model_loaded_PF_SMAP.eval()
nx = (forcing.shape[-1] + rawData.shape[-1], rawData.shape[-1], len(varConst_Noah))
ny = target.shape[-1]

totalTrainMeanLst = list()
totalTestMeanLst = list()
totalTrainMedianLst = list()
totalTestMedianLst = list()

# ======= dPL =======
saveEpoch=100
minibatch=[300, 240]  ## minibatch[0] is batch size, minibatch[1] is the length of the training instances

print('On dPL..')
## save dPL path
path_Inv = cDir + '/dPL_models/model_dPL_'\
           + subsetLst[0] + '_hs' + str(hiddenSizeLst[0]) + '_tr' + str(trainYear) + \
           '_SMAP_PM_L3E'
outFolder = os.path.join(cDir, path_Inv)
if os.path.exists(outFolder) is False:
    os.mkdir(outFolder)

## dPL setting
model_Inv = rnn.CudnnLstmModel_Inv(nx=nx, ny=ny, hiddenSize=hiddenSizeLst[0], filename=filename)

## loss function
lossFun_Inv1 = crit.RmseLoss()

lossFun_Inv2 = crit.RangeBoundLoss(lb=lb, ub=ub, factor=1)
lossFun_Inv = crit.sumOfLoss(lossFun_Inv1, lossFun_Inv2)

## ========== training model

## scaling transfer
scaleTrans = True

##### train ####
### gZ
model_Inv = train.trainModel(
    model_Inv, (forcing, target_GF), modelTarget, rawData, lossFun_Inv, nEpoch=epoch_dPL, miniBatch=minibatch,\
        optLr=0.5, optRho=0.95, optWd=0.00001, saveEpoch=saveEpoch, saveFolder=outFolder, lyrMode=0, scaleTrans=True)

#### gA
# model_Inv = train.trainModel(
#     model_Inv, (forcing, target_GF), modelTarget, rawData, lossFun_Inv, nEpoch=epoch_dPL, miniBatch=minibatch,\
#         optLr=0.5, optRho=0.95, optWd=0.00001, saveEpoch=saveEpoch, saveFolder=outFolder, lyrMode='PureRaw', scaleTrans=True)

## save trained model
train.saveModel(outFolder, model_Inv, epoch_dPL, modelName=modelName)

