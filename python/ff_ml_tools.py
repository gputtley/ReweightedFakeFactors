import ROOT
import uproot
import numpy as np
import pandas as pd
import re
import warnings
import re
import pickle
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import fnmatch
import itertools
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from numpy.lib.recfunctions import append_fields
from pandas import DataFrame, RangeIndex
from root_numpy import root2array, list_trees
from root_numpy import list_branches
from root_numpy.extern.six import string_types
from math import ceil
from hep_ml.metrics_utils import ks_2samp_weighted


def setup_dataframe(loc,root_files,channel,year,vars,selection):
  # Additional functions to work with
  func_dict = {"fabs":"abs(x)","cos":"math.cos(x)","sin":"math.sin(x)", "cosh":"math.cosh(x)", "sinh":"math.sinh(x)", "ln":"math.log(x)"}

  # Set up variables to get from root file trees
  get_variables = []

  # Get variables needed for training
  mod_vars = []
  delim=["*","/","(",")","==",">=","<",">","&&","||"]
  for i in vars:
    split_list = custom_split(i.replace(" ",""),deliminators=delim)
    if len(split_list) == 1:
      get_variables.append(i)
    else:
      i_strip = i.replace(" ","")
      mod_vars.append(i_strip)
      for j in split_list:
        if j not in get_variables and not j.replace(".","").isdigit() and not j in delim and not j in list(func_dict.keys()): get_variables.append(j)

  # Get variables needed for selection
  split_list = custom_split(selection.replace(" ",""),deliminators=delim)
  for j in split_list:
    if j not in get_variables and not j.replace(".","").isdigit() and not j in delim and not j in list(func_dict.keys()): get_variables.append(j)
  #print ">> Getting variables from ROOT files: "
  #print get_variables

  # Add all root file trees for variables determined above to dataframes
  if str(type(i)) == "<type 'str'>": root_files = [root_files]
  for i in root_files:
    tree = uproot.open(loc+'/'+i+'_'+channel+'_'+year+'.root')["ntuple"]
    if i == root_files[0]:
      df = tree.pandas.df(get_variables)
    else:
      df = pd.concat([df,tree.pandas.df(get_variables)],ignore_index=True, sort=False)
  # Cut dataframe with selection
  #print ">> Selected events matching the condition: "
  #print selection

  # Rename variables and deliminators
  split_list = custom_split(selection.replace(" ",""),deliminators=["*","/","(",")","==",">=","<",">","&&","||"])
  for j in range(0,len(split_list)):
    if not split_list[j].replace(".","").isdigit() and not split_list[j] in delim and not split_list[j] in list(func_dict.keys()): 
      split_list[j] = '(df["{}"]'.format(split_list[j])
    elif split_list[j] == "&&":
      split_list[j] = ")&"
    elif split_list[j] == "||":
      split_list[j] = ")|"
  split_list.append(")")

  # Rename functions
  for func_name, func_def in func_dict.items():
    for i in find_in_list(func_name,split_list):
      split_list[find_closed_bracket(i+1,split_list)] = ".apply(lambda x: {})".format(func_def) + split_list[find_closed_bracket(i+1,split_list)]
    for j in split_list:
      if j == func_name: split_list.remove(j)

  # Combine command and cut dataframe
  cmd = ""
  for i in split_list: cmd += i
  df = eval('df[{}]'.format(cmd))

  # Calculate modified variables
  #print ">> Calculating modified variables and adding them to the dataframe"
  for i in mod_vars:
    var_string = i.replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")","")
    split_list = custom_split(i.replace(" ",""),deliminators=delim)
    for j in range(len(split_list)-1,-1,-1):
      if "/" in split_list[j]:
        split_list[j] = ".divide("
        split_list[j+1] += ")"
      elif "*" in split_list[j]:
        split_list[j] = ".multiply("
        split_list[j+1] += ")"
      elif not split_list[j].replace(".","").isdigit() and not split_list[j] in delim and not split_list[j] in list(func_dict.keys()) and not split_list[j]=="":
        split_list[j] =  "df.loc[:,'{}']".format(split_list[j])
      elif "==" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x=={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif ">=" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x>={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif "<=" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x<={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif ">" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x>{} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif "<" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x<{} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      else:
        for func_name, func_def in func_dict.items():
          if func_name in split_list[j]:
            split_list[j] = "{}.apply(lambda x: {})".format(split_list[j+2],func_def)
            split_list[j+1] = ""
            split_list[j+2] = ""
            split_list[j+3] = ""
    cmd = ""
    for i in split_list: cmd += i
    df.loc[:,var_string] = eval(cmd)
       
  # Removing variables not used in training
  #print ">> Removing variables that are not needed"
  for i in get_variables:
    if i not in vars:
      var_name  = i.replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")","")
      df = df.drop([var_name],axis=1)

  return df

def custom_split(string,deliminators=[","]):
  single_deliminators = []
  plus_deliminators = []
  for i in deliminators:
    if len(i) == 1: 
      single_deliminators.append(i)
    else:
      for j in range(0,len(i)):
        if j == 0:
          plus_deliminators.append(i[j])
        else:
          plus_deliminators.append(plus_deliminators[len(plus_deliminators)-1]+i[j])

  out_list = []
  for i in range(0,len(string)):
    if string[i] in single_deliminators:
      out_list.append(string[i])
    elif len(out_list) == 0:
      out_list.append(string[i])
    elif string[i-1]+string[i] in plus_deliminators: 
      out_list[len(out_list)-1] += string[i]
    elif string[i-1] in single_deliminators:
      out_list.append(string[i])
    elif string[i] in plus_deliminators and out_list[-1] not in plus_deliminators:
      out_list.append(string[i])
    elif out_list[-1] in deliminators:
      out_list.append(string[i])
    else:
      out_list[len(out_list)-1] += string[i]

  return out_list

def find_in_list(find,flist):
  find_list = []
  for i in range(0,len(flist)):
    if flist[i] == find: find_list.append(i)
  return find_list

def find_closed_bracket(open_bracket,flist):
  ob = 1
  for i in range(open_bracket+1,len(flist)):
    if flist[i] == "(": ob += 1
    elif flist[i] == ")": ob -= 1
    if ob == 0: break
  return i

def to_root(df, path, key='my_ttree', mode='w', store_index=True, *args, **kwargs):
    """
    Write DataFrame to a ROOT file.
    Parameters
    ----------
    path: string
        File path to new ROOT file (will be overwritten)
    key: string
        Name of tree that the DataFrame will be saved as
    mode: string, {'w', 'a'}
        Mode that the file should be opened in (default: 'w')
    store_index: bool (optional, default: True)
        Whether the index of the DataFrame should be stored as
        an __index__* branch in the tree
    Notes
    -----
    Further *args and *kwargs are passed to root_numpy's array2root.
    >>> df = DataFrame({'x': [1,2,3], 'y': [4,5,6]})
    >>> df.to_root('test.root')
    The DataFrame index will be saved as a branch called '__index__*',
    where * is the name of the index in the original DataFrame
    """

    if mode == 'a':
        mode = 'update'
    elif mode == 'w':
        mode = 'recreate'
    else:
        raise ValueError('Unknown mode: {}. Must be "a" or "w".'.format(mode))

    column_name_counts = Counter(df.columns)
    if max(column_name_counts.values()) > 1:
        raise ValueError('DataFrame contains duplicated column names: ' +
                         ' '.join({k for k, v in column_name_counts.items() if v > 1}))

    from root_numpy import array2tree
    # We don't want to modify the user's DataFrame here, so we make a shallow copy
    df_ = df.copy(deep=False)

    if store_index:
        name = df_.index.name
        if name is None:
            # Handle the case where the index has no name
            name = ''
        df_['__index__' + name] = df_.index

    # Convert categorical columns into something root_numpy can serialise
    for col in df_.select_dtypes(['category']).columns:
        name_components = ['__rpCaT', col, str(df_[col].cat.ordered)]
        name_components.extend(df_[col].cat.categories)
        if ['*' not in c for c in name_components]:
            sep = '*'
        else:
            raise ValueError('Unable to find suitable separator for columns')
        df_[col] = df_[col].cat.codes
        df_.rename(index=str, columns={col: sep.join(name_components)}, inplace=True)

    arr = df_.to_records(index=False)

    root_file = ROOT.TFile.Open(path, mode)
    if not root_file:
        raise IOError("cannot open file {0}".format(path))
    if not root_file.IsWritable():
        raise IOError("file {0} is not writable".format(path))

    # Navigate to the requested directory
    open_dirs = [root_file]
    for dir_name in key.split('/')[:-1]:
        current_dir = open_dirs[-1].Get(dir_name)
        if not current_dir:
            current_dir = open_dirs[-1].mkdir(dir_name)
        current_dir.cd()
        open_dirs.append(current_dir)

    # The key is now just the top component
    key = key.split('/')[-1]

    # If a tree with that name exists, we want to update it
    tree = open_dirs[-1].Get(key)
    if not tree:
        tree = None
    tree = array2tree(arr, name=key, tree=tree)
    tree.Write(key, ROOT.TFile.kOverwrite)
    root_file.Close()

def draw_distributions(original, target, original_weights, target_weights, columns=['mt_tot'], new_original_weights=None):
  plt.figure(figsize=[15, 12])
  for id, column in enumerate(columns, 1):
    xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
    plt.subplot(3, 3, id)
    if isinstance(new_original_weights, pd.Series):
      plt.hist(original[column], weights=new_original_weights.multiply(original_weights), range=xlim, bins=50, alpha=0.8, color="red", label="original")
    else:
      plt.hist(original[column], weights=original_weights, range=xlim, bins=50, alpha=0.8, color="red", label="original")
    plt.hist(target[column], weights=target_weights, range=xlim, bins=50, alpha=0.8, color="blue", label="target")
    plt.title(column)
    plt.legend()
  plt.show()

def KS_test(original, target, original_weights, target_weights, columns=['mt_tot'], new_original_weights=None):
  ks_dict = {}
  ks_total = 0
  for id, column in enumerate(columns, 1):
    if isinstance(new_original_weights, pd.Series):
      ks_dict[column] = round(ks_2samp_weighted(original[column], target[column], weights1=new_original_weights.multiply(original_weights), weights2=target_weights),6)
    else:
      ks_dict[column] = round(ks_2samp_weighted(original[column], target[column], weights1=original_weights, weights2=target_weights),6)
    ks_total += ks_dict[column]
  return round(ks_total,6), ks_dict

def SelectColumns(df,columns):
  # Get modified variables
  mod_vars = []
  new_df = df.copy(deep=True)
  orig_keys = new_df.keys()
  delim=["*","/","(",")","==",">=","<",">","&&","||"]
  for i in columns:
    split_list = custom_split(i.replace(" ",""),deliminators=delim)
    if len(split_list) > 1:
      mod_vars.append(i.replace(" ",""))
  func_dict = {"fabs":"abs(x)","cos":"math.cos(x)","sin":"math.sin(x)", "cosh":"math.cosh(x)", "sinh":"math.sinh(x)", "ln":"math.log(x)"}
  # Calculate modified variables
  for i in mod_vars:
    var_string = i.replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")","")
    split_list = custom_split(i.replace(" ",""),deliminators=delim)
    for j in range(len(split_list)-1,-1,-1):
      if "/" in split_list[j]:
        split_list[j] = ".divide("
        split_list[j+1] += ")"
      elif "*" in split_list[j]:
        split_list[j] = ".multiply("
        split_list[j+1] += ")"
      elif not split_list[j].replace(".","").isdigit() and not split_list[j] in delim and not split_list[j] in list(func_dict.keys()) and not split_list[j]=="":
        split_list[j] =  "new_df.loc[:,'{}']".format(split_list[j])
      elif "==" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x=={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif ">=" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x>={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif "<=" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x<={} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif ">" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x>{} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      elif "<" in split_list[j]:
        split_list[j] = ".apply(lambda x: 1 if x<{} else 0)".format(split_list[j+1])
        split_list[j+1] = ""
      else:
        for func_name, func_def in func_dict.items():
          if func_name in split_list[j]:
            split_list[j] = "{}.apply(lambda x: {})".format(split_list[j+2],func_def)
            split_list[j+1] = ""
            split_list[j+2] = ""
            split_list[j+3] = ""
    cmd = ""
    for i in split_list: cmd += i
    new_df.loc[:,var_string] = eval(cmd)

  # rename X_vars
  columns_string = []
  for i in range(0,len(columns)):
    columns_string.append(columns[i].replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")",""))


  # Removing variables not used
  for i in orig_keys:
    if i not in columns_string:
      var_name  = i.replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")","")
      new_df = new_df.drop([var_name],axis=1)

  new_df = new_df.loc[:,columns_string]
  return new_df

def GetDataframe(baseline,lumi,params,input_folder,files,channel,year,X_vars,weights,y_var,selection_vars,scoring_vars,other_vars,data=False):
  for i in files:
    if channel in ["et","mt"]:
      df =  setup_dataframe(input_folder,i,channel,year,list(dict.fromkeys(X_vars+weights+[y_var.replace("X","2")]+selection_vars+scoring_vars+other_vars)),baseline)
    elif channel == "tt":
      df =  setup_dataframe(input_folder,i,channel,year,list(dict.fromkeys(X_vars+weights+[y_var.replace("X","1"),y_var.replace("X","2")]+selection_vars+scoring_vars+other_vars)),baseline)
    if len(df>0): # needed if no events meet selection
      if not data:
        df.loc[:,"scale"] = lumi*params[i]['xs']/params[i]['evt']
      else:
        df.loc[:,"scale"] = 1
      for wt in weights:
        df.loc[:,"scale"] = df.loc[:,"scale"].multiply(df.loc[:,wt])
      df = df.drop(weights,axis=1)
    if i == files[0]:
      df_total = df
    else:
      df_total = pd.concat([df,df_total],ignore_index=True, sort=False)
    del df
  return df_total

def CutAndScale(df,selection,scale):
  if isinstance(df, list):
    df = pd.concat(df,ignore_index=True, sort=False)
  delim=["*","/","(",")","==",">=","<",">","&&","||"]
  func_dict = func_dict = {"fabs":"abs(x)","cos":"math.cos(x)","sin":"math.sin(x)", "cosh":"math.cosh(x)", "sinh":"math.sinh(x)", "ln":"math.log(x)"}
  # Rename variables and deliminators
  split_list = custom_split(selection.replace(" ",""),deliminators=delim)
  for j in range(0,len(split_list)):
    if not split_list[j].replace(".","").isdigit() and not split_list[j] in delim and not split_list[j] in list(func_dict.keys()):
      split_list[j] = '(df["{}"]'.format(split_list[j])
    elif split_list[j] == "&&":
      split_list[j] = ")&"
    elif split_list[j] == "||":
      split_list[j] = ")|"
  split_list.append(")")

  # Rename functions
  for func_name, func_def in func_dict.items():
    for i in find_in_list(func_name,split_list):
      split_list[find_closed_bracket(i+1,split_list)] = ".apply(lambda x: {})".format(func_def) + split_list[find_closed_bracket(i+1,split_list)]
    for j in split_list:
      if j == func_name: split_list.remove(j)

  # Combine command and cut dataframe
  cmd = ""
  for i in split_list: cmd += i
  df = eval('df[{}]'.format(cmd))
  df.loc[:,"scale"] = scale*df.loc[:,"scale"]
  return df
  
def SetUpDataframeDict(data,X_vars,y_var,selection_vars,scoring_vars,other_vars):
  # drop selection variables
  data = data.drop(selection_vars,axis=1)

  # set up train and test dataset
  X, y = data.drop([y_var],axis=1),data.loc[:,y_var]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
  train = pd.concat([X_train, y_train], axis=1)
  test = pd.concat([X_test, y_test], axis=1)

  for i in scoring_vars:
    if i not in X_vars:
      other_vars.append(i)

  original_train, target_train = train[(train[y_var]==0)], train[(train[y_var]==1)]
  original_test, target_test = test[(test[y_var]==0)], test[(test[y_var]==1)]

  # rename X_vars
  for i in range(0,len(X_vars)):
    X_vars[i] = X_vars[i].replace("*","_times_").replace("/","_over_").replace("==","_").replace(">","_gt_").replace("<","_lt_").replace("(","_").replace(")","")

  dfs = {}

  dfs["original_weights_train"], dfs["target_weights_train"] = original_train.loc[:,"scale"], target_train.loc[:,"scale"]
  dfs["original_other_train"], dfs["target_other_train"] = original_train.loc[:,other_vars], target_train.loc[:,other_vars]
  #dfs["original_train"], dfs["target_train"] = original_train.drop([y_var,"scale"]+other_vars,axis=1), target_train.drop([y_var,"scale"]+other_vars,axis=1)
  dfs["original_train"], dfs["target_train"] = original_train.loc[:,X_vars], target_train.loc[:,X_vars]

  dfs["original_weights_test"], dfs["target_weights_test"] = original_test.loc[:,"scale"], target_test.loc[:,"scale"]
  dfs["original_other_test"], dfs["target_other_test"] = original_test.loc[:,other_vars], target_test.loc[:,other_vars]
  #dfs["original_test"], dfs["target_test"] = original_test.drop([y_var,"scale"]+other_vars,axis=1), target_test.drop([y_var,"scale"]+other_vars,axis=1)
  dfs["original_test"], dfs["target_test"] = original_test.loc[:,X_vars], target_test.loc[:,X_vars]

  return dfs

def GetNormalisation(dfs,model):
  gb_weights_train = model.predict_weights(dfs["original_train"],dfs["original_weights_train"],merge_weights=False)
  wts_1 = dfs["original_weights_train"].multiply(gb_weights_train)
  norm = dfs["target_weights_train"].sum()/wts_1.sum()
  return norm

def ScoreModel(dfs,model,X_vars,scoring_vars,other_vars,silent=False):
 # Applying model to train dataset
  new_dfs = {}
  for key,val in dfs.items():
    new_dfs[key] = val.copy(deep=True)
  gb_weights_train = model.predict_weights(dfs["original_train"],dfs["original_weights_train"],merge_weights=False)
  wts_1 = dfs["original_weights_train"].multiply(gb_weights_train)

  for i in scoring_vars:
    if i not in X_vars:
      other_vars.append(i)

  # add variables back that are needed for scoring
  for i in other_vars:
    new_dfs["original_train"].loc[:,i] = dfs["original_other_train"].loc[:,i]
    new_dfs["target_train"].loc[:,i] = dfs["target_other_train"].loc[:,i]

  if not silent:
    print "----------------------------------------------------------------------------------------------"
    print "Train dataset"
    print "----------------------------------------------------------------------------------------------"
    print KS_test(new_dfs["original_train"], new_dfs["target_train"], wts_1, new_dfs["target_weights_train"], columns=scoring_vars)

  gb_weights_test = model.predict_weights(new_dfs["original_test"],new_dfs["original_weights_test"],merge_weights=False)
  wts_2 = new_dfs["original_weights_test"].multiply(gb_weights_test)

  for i in scoring_vars:
    if i not in X_vars:
      other_vars.append(i)

  # add variables back that are needed for scoring
  for i in other_vars:
    new_dfs["original_test"].loc[:,i] = dfs["original_other_test"].loc[:,i]
    new_dfs["target_test"].loc[:,i] = dfs["target_other_test"].loc[:,i]
  
  score =  KS_test(new_dfs["original_test"], new_dfs["target_test"], wts_2, new_dfs["target_weights_test"], columns=scoring_vars)
  if not silent:
    print "----------------------------------------------------------------------------------------------"
    print "Test dataset"
    print "----------------------------------------------------------------------------------------------"
    print score
  return score


def CreateBatchJob(name,cmssw_base,cmd_list):
  if os.path.exists(name): os.system('rm %(name)s' % vars())
  os.system('echo "#!/bin/bash" >> %(name)s' % vars())
  os.system('echo "cd %(cmssw_base)s/src/UserCode/ICHiggsTauTau/ReweightedFakeFactors" >> %(name)s' % vars())
  os.system('echo "source /vols/grid/cms/setup.sh" >> %(name)s' % vars())
  os.system('echo "export SCRAM_ARCH=slc6_amd64_gcc481" >> %(name)s' % vars())
  os.system('echo "eval \'scramv1 runtime -sh\'" >> %(name)s' % vars())
  os.system('echo "ulimit -s unlimited" >> %(name)s' % vars())
  for cmd in cmd_list:
    os.system('echo "%(cmd)s" >> %(name)s' % vars())
  os.system('chmod +x %(name)s' % vars())
  print "Created job:",name

def SubmitBatchJob(name,time=180,memory=24,cores=1):
  error_log = name.replace('.sh','_error.log')
  output_log = name.replace('.sh','_output.log')
  if os.path.exists(error_log): os.system('rm %(error_log)s' % vars())
  if os.path.exists(output_log): os.system('rm %(output_log)s' % vars())
  if cores>1: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -pe hep.pe %(cores)s -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())
  else: os.system('qsub -e %(error_log)s -o %(output_log)s -V -q hep.q -l h_rt=0:%(time)s:0 -l h_vmem=%(memory)sG -cwd %(name)s' % vars())
