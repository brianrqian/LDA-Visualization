#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import os
import json
import pandas as pd

sys.path.insert(0, 'src/test')
sys.path.insert(0, 'src')
import LDA

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'.

    `main` runs the targets in order of data=>analysis=>model.
    '''
    if 'test' in targets:
        datafile = 'test/test.csv'
        hdsi = 'HDSI.csv'
        LDA.LDA_model(datafile, hdsi)
    #with open('test-params.json') as fh:
    #    data_cfg = json.load(fh)
    #with open('hdsi-params.json') as fh:
    #    hdsi_cfg = json.load(fh)
    #LDA.LDA_model(**data_cfg, **hdsi_cfg)
    else:
        datafile = 'final_hdsi_faculty_updated.csv'
        hdsi = 'HDSI.csv'
        LDA.LDA_model(datafile, hdsi)

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)


# In[ ]:
