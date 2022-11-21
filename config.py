#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:13:58 2021

@author: frederictheunissen
"""
import pickle as pk

# Load the data base
rootPath = '/Users/frederictheunissen/Code/songephys/'
dataPath = 'data/birds/'
outPath =  rootPath + dataPath + 'HerminaDataBase.pkl'

# rootPath = '/Users/frederictheunissen/Google Drive/My Drive/julie/'
# outPath = rootPath+'JulieDataBase.pkl'

fileIn = open(outPath,"rb")
DFAll = pk.load(fileIn)
#DF = DFAll[DFAll['call'] != 'Wh']
DF = DFAll

fileIn.close() 
