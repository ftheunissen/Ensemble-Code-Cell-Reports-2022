#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:36:09 2021

@author: frederictheunissen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

import config



def naiveBayes(unitChoice, plotFlg = False, usingAllCombos = False, code='PC', verbose = False, nperm=100):
    # naiveBayes runs a guassian classifier for call types using the responses in the units in the list unitChoice
    
    dfDataBase = config.DF
    
    # Number of units
    numUnits = len(unitChoice)
    
    # Number of PCs (for PC code)
    nPCs = dfDataBase['PC'][0].shape[0]
    
    # Find the unique stims played for the chosen unit
    unitIndexDict = dict()
    unitIndexLen = []
    stimList = []
    for unit in unitChoice:   
        unitIndex = dfDataBase.index[dfDataBase['site']+'_'+ dfDataBase['unit'] == unit]
        unitIndexDict[unit] = unitIndex
        unitIndexLen.append(len(unitIndex))
        stimList.extend(list(dfDataBase['stim'][unitIndex].array))
    
    stimNames = np.unique(stimList)
    
    # Generate ensemble responses  for the units in unitChoice
    
    for iperm in range(nperm):
        # Training Trials
        rowListList = []
        totTrials = 0

        # Testing Trials
        rowListListTest = []
        totTestTrials = 0

        for stim in stimNames:
            goodStim = 2    # This means that we have at least two trials for each stim for all units
        
            # Check to see if we have enough trials for all units for this stim
            for unit in unitChoice:
                unitIndex = unitIndexDict[unit]
                unitIndexStim = [ind for ind in unitIndex if dfDataBase['stim'][ind] == stim]
                if len(unitIndexStim) == 0:
                    goodStim = 0
                    print('Stim ', stim, 'has no response for unit', unit, '. Skipping')
                    break
                elif len(unitIndexStim) == 1:
                    goodStim = 1
            
            # If so continue
            if goodStim > 0 :
                rowList2Cat = []
                rowList2CatLen = []
                for unit in unitChoice:
                    unitIndex = unitIndexDict[unit]
                    unitIndexStim = [ind for ind in unitIndex if dfDataBase['stim'][ind] == stim]
                    rowList2Cat.append(unitIndexStim)
                    rowList2CatLen.append(len(unitIndexStim))
            
                if goodStim > 1 :  # Use one of the responses for the test set
                    listTest = []
                    for iunit, unit in enumerate(unitChoice):
                        rowchosen = np.random.choice(rowList2Cat[iunit], 1, replace=False)
                        listTest.append(rowchosen[0])
                        rowList2Cat[iunit].remove(rowchosen)
                        rowList2CatLen[iunit] = len(rowList2Cat[iunit])
                    rowListListTest.append(listTest)
                    totTestTrials +=1
            
                if usingAllCombos:
                    numSamp = np.prod(rowList2CatLen)
                    totTrials += numSamp
            
                    list_of_lists = []

                    indUnit = np.zeros((numUnits,), dtype=int)
                    indUnit[0] = -1
            
                    for i in range(numSamp):
                        for iunit, unit in enumerate(unitChoice):
                            indUnit[iunit] += 1
                            if (indUnit[iunit] >= len(rowList2Cat[iunit]) ):
                                indUnit[iunit] = 0
                            else: 
                                break
                        trialList = []
                        for iunit, unit in enumerate(unitChoice):
                            trialList.append(rowList2Cat[iunit][indUnit[iunit]])
                
                        list_of_lists.append(trialList)
                else:
                    numSamp = np.max(rowList2CatLen)
                    totTrials += numSamp
                    
                    list_of_lists = []
                    for i in range(numSamp):
                        trialList = []
                        for iunit, unit in enumerate(unitChoice):
                            if i < len(rowList2Cat[iunit]):
                                trialList.append(rowList2Cat[iunit][i])
                            else:
                                trialList.append(np.random.choice(rowList2Cat[iunit], 1, replace=False)[0])
                        list_of_lists.append(trialList)
        
                    rowListList.extend(list_of_lists)
                    
        # If this combinationyields less than 10 Testtrial - continue
        if totTestTrials < 2:
            iperm -= 1
            print('Permutation:', iperm+1)
            print('Total number of trials:', totTrials)
            print('Total number of test trials:', totTestTrials)
            continue

        if verbose:        
            print('Total number of trials:', totTrials)
            print('Total number of test trials:', totTestTrials)
    
        # Make X (neural code), Y (call type)
        if code == 'z':
            Xfit = np.zeros((totTrials, numUnits*1))
            Xtest = np.zeros((totTestTrials, numUnits*1))
        elif code == 'PC':
            Xfit = np.zeros((totTrials, numUnits*(1+nPCs)))
            Xtest = np.zeros((totTestTrials, numUnits*(1+nPCs)))       
        Yfit = []
        Ytest = []
        
        for irow,rowList in enumerate(rowListList):
            if code == 'z':
                Xfit[irow,:] = dfDataBase['z'][rowList].array
            elif code == 'PC':
                Xfit[irow,:] = np.hstack([np.hstack((dfDataBase['z'][ind], dfDataBase['PC'][ind])) for ind in rowList])
            
            Yfit.append(np.unique(dfDataBase['call'][rowList].array)[0])
        Yfit = np.array(Yfit)

    
        
        for irow,rowList in enumerate(rowListListTest):
            if code == 'z':
                Xtest[irow,:] = dfDataBase['z'][rowList].array
            elif code == 'PC':
                Xtest[irow,:] = np.hstack([np.hstack((dfDataBase['z'][ind], dfDataBase['PC'][ind])) for ind in rowList])
            Ytest.append(np.unique(dfDataBase['call'][rowList].array)[0])
    
        Ytest = np.array(Ytest)
    
    
        # Fit and test the classifier clf
        nClasses = len(np.unique(Yfit))
        if nClasses != 10 :  #  This is  bad cod because it has to be changed for NW vs all calls
            print('Insufficient data')
            continue

        clf = GaussianNB(priors=np.ones((nClasses,))/nClasses)
        clf.fit(Xfit,Yfit)

        probTest = clf.predict_proba(Xtest)

        # Make a confusion matrix
        confMat = np.zeros((nClasses,nClasses))
        testsPerClass = np.zeros((nClasses,))

        for i,y in enumerate(Ytest):
            classID = np.argwhere(clf.classes_ == y)
            testsPerClass[classID] += 1    
            confMat[classID,:] += probTest[i]
            
        # Copy data
        if iperm == 0:
            confMatAll = np.copy(confMat)
            testsPerClassAll = np.copy(testsPerClass)
        else:
            confMatAll += confMat
            testsPerClassAll += testsPerClass
    
    # The confusion matrix is not normalized so that we can average it correctly later on.. 
    #for i in range(nClasses):
    #    confMat[i,:] /= testsPerClass[i]
    
    # Print percent correct classification by taking average of diagonal of confusion matrix.
    sumCorrect = np.trace(confMatAll)
    sumTest = np.sum(testsPerClassAll)
    pcc = 100.0*(sumCorrect/sumTest)
    
    if verbose:
        print('PCC %.0f %%' % (pcc))
    
    if plotFlg:
        # Normalize confMat for display
        confMatNorm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            if testsPerClass[i] > 0 :
                confMatNorm[i,:] = confMatAll[i,:]/testsPerClassAll[i]
                
        # Display comfusion matrix
        cmap='viridis'
        fig, ax = plt.subplots()
        im_ = ax.imshow(confMatNorm*100.0, interpolation='nearest', cmap=cmap)

        fig.colorbar(im_, ax=ax)
        ax.set(xticks=np.arange(nClasses),
               yticks=np.arange(nClasses),
               xticklabels=clf.classes_,
               yticklabels=clf.classes_,
               ylabel="Actual Call Type",
               xlabel="Predicted Call Type",
               title='PPC %.0f %%' % (pcc))
    
    return pcc, confMatAll, testsPerClassAll

def naiveBayesFET(unitChoice, plotFlg = False, code='PC', verbose = False, weightedVote = False):
    # naiveBayes runs a guassian classifier for call types using the responses in the units in the list unitChoice
    # The FET version calculates the parameters of the gaussian for each unit.
    # I tried one version with a weighted sum of probabilities that failed.  That code is active when weightedVote is set to true.
    # Noted differences with scikit-learn: 
    # 1. FET uses the unbiased estimate of the variance of the guassian, scikit uses the ML estimate (which is biased)
    # 2. scikit adds an epsilon to this variance wich is 1e-9*max of variance
    
    global dfDataBase
    # Number of units
    numUnits = len(unitChoice)    
    # Number of PCs (for PC code)
    nPCs = dfDataBase['PC'][0].shape[0]
    # Number of calls is ncalls with names callNames
        
    # Find the unique stims played for the chosen unit
    unitIndexDict = dict()
    unitIndexLen = []
    stimList = []
    for unit in unitChoice:   
        unitIndex = dfDataBase.index[dfDataBase['site']+'_'+ dfDataBase['unit'] == unit]
        unitIndexDict[unit] = unitIndex
        unitIndexLen.append(len(unitIndex))
        stimList.extend(list(dfDataBase['stim'][unitIndex].array))
    
    stimNames = np.unique(stimList)
    
    # Find testing and training trials for each stim

    # Training data for each unit
    unitRowList = [[]] * numUnits

    # Testing Trials - One for each stim
    rowListListTest = []
    totTestTrials = 0

    for stim in stimNames:
        goodStim = 2    # This means that we have at least two trials for each stim for all units
        
        # Check to see if we have enough trials for all units for this stim
        for unit in unitChoice:
            unitIndex = unitIndexDict[unit]
            unitIndexStim = [ind for ind in unitIndex if dfDataBase['stim'][ind] == stim]
            if len(unitIndexStim) == 0:
                goodStim = 0
                print('Stim ', stim, 'has no response for unit', unit, '. Skipping')
                break
            elif len(unitIndexStim) == 1:
                goodStim = 1
            
        # If so continue
        if goodStim > 0 :
            rowList2Cat = []
            for unit in unitChoice:
                unitIndex = unitIndexDict[unit]
                unitIndexStim = [ind for ind in unitIndex if dfDataBase['stim'][ind] == stim]
                rowList2Cat.append(unitIndexStim)
                      
            listTest = []
            for iunit, unit in enumerate(unitChoice):
                rowchosen = np.random.choice(rowList2Cat[iunit], 1, replace=False)
                listTest.append(rowchosen[0])
                rowList2Cat[iunit].remove(rowchosen)
                unitRowList[iunit].extend(rowList2Cat[iunit])
            
            rowListListTest.append(listTest)
            totTestTrials +=1
    
    # For each unit generate mean, variance, and number of trials for each call type.
    unitGaussianModelList = [None]*numUnits
    for iunit, unit in enumerate(unitChoice):
        unitGaussianModelList[iunit] = [None]*ncalls 
        for icall, call in enumerate(callNames):
            unitModel = dict()
            unitModel['count'] = 0
            unitModel['call'] = call
            
            # Rows for this unit and call:
            rowList = [ind for ind in unitRowList[iunit] if dfDataBase['call'][ind] == call]
            
            if len(rowList) > 2:
                if code == 'z':
                    unitModel['means'] = np.mean(dfDataBase['z'][rowList].values, axis = 0)
                    unitModel['stds'] = np.std(dfDataBase['z'][rowList].values, axis = 0, ddof = 1)
                elif code == 'PC':
                    unitModel['means'] = np.hstack((np.mean(dfDataBase['z'][rowList].values, axis = 0), np.mean(dfDataBase['PC'][rowList].values, axis = 0)))
                    unitModel['stds'] = np.hstack((np.std(dfDataBase['z'][rowList].values, axis = 0, ddof=1), np.std(dfDataBase['PC'][rowList].values, axis = 0, ddof = 1)))

                unitModel['count'] = len(rowList) - 2 
            
            unitGaussianModelList[iunit][icall] = unitModel
  
    # Calculate test probabilities  
    confMat = np.zeros((ncalls,ncalls))
    testsPerClass = np.zeros((ncalls,))
    
    for irow,rowList in enumerate(rowListListTest):

        ytest = np.unique(dfDataBase['call'][rowList].array)[0]
        probCall = np.zeros(ncalls)
        if verbose:
            print('++++++++++++++++++++++++ Call test', ytest, '++++++++++++++++++++++')
            print(callNames)

        for iunit, unit in enumerate(unitChoice):
            unitVotes = 0
            probUnit = np.zeros(ncalls)
            if code == 'z':
                xtest = dfDataBase['z'][rowList[iunit]]
            elif code == 'PC':
                xtest = np.hstack((dfDataBase['z'][rowList[iunit]], dfDataBase['PC'][rowList[iunit]]))
            for icall, call in enumerate(callNames):
                unitModel = unitGaussianModelList[iunit][icall]
                unitVotes += unitModel['count']
                if unitModel['count']:
                    probval = 0
                    for i, xval in enumerate(xtest):
                        probval += norm.logpdf(xval, loc = unitModel['means'][i], scale =  unitModel['stds'][i])
                    probUnit[icall] = np.exp(probval)
                else:
                    probUnit[icall] = np.nan
            
            # Fix nans
            meanProb = np.nanmean(probUnit)
            for icall, call in enumerate(callNames):
                if np.isnan(probUnit[icall]):
                    probUnit[icall] = meanProb
                    
            # Calculate contribution of this unit
            if weightedVote:
                sumProb = np.sum(probUnit)
                probUnit = probUnit*unitVotes/sumProb
                probCall += probUnit
            else:
                probUnit = np.log(probUnit)
                probCall += probUnit
            
            if verbose:
                print('\n----Adding unit', iunit)
                print(probUnit)
                print(probCall)

                          
        # Final normalization
        if ~weightedVote:
            probCall = np.exp(probCall-np.mean(probCall))
        probCall = probCall/np.sum(probCall)

        
        # Fill in the data
        classID = np.argwhere(callNames == ytest)[0]
        testsPerClass[classID] += 1    
        confMat[classID,:] += probCall
        if verbose:
            print('\n------- FINAL')
            print(ytest, classID)
            print(callNames)
            print(probCall)
            print(confMat[classID,:])
        
        
    # Print percent correct classification by taking average of diagonal of confusion matrix.
    sumCorrect = np.trace(confMat)
    sumTest = np.sum(testsPerClass)
    pcc = 100.0*(sumCorrect/sumTest)
    
    if verbose:
        print('PCC %.0f %%' % (pcc))
    
    if plotFlg:
        # Normalize confMat for display
        confMatNorm = np.zeros((ncalls,ncalls))
        for i in range(ncalls):
            if testsPerClass[i] > 0 :
                confMatNorm[i,:] = confMat[i,:]/testsPerClass[i]
                
        # Display comfusion matrix
        cmap='viridis'
        fig, ax = plt.subplots()
        im_ = ax.imshow(confMatNorm*100.0, interpolation='nearest', cmap=cmap)

        fig.colorbar(im_, ax=ax)
        ax.set(xticks=np.arange(ncalls),
               yticks=np.arange(ncalls),
               xticklabels=callNames,
               yticklabels=callNames,
               ylabel="Actual Call Type",
               xlabel="Predicted Call Type",
               title='PPC %.0f %%' % (pcc))
    
    return pcc, confMat, testsPerClass
