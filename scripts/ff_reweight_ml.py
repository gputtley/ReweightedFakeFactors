import argparse
import json
import root_numpy
import pandas
import numpy
import pickle
import json
from UserCode.ReweightedFakeFactors import reweight
from UserCode.ReweightedFakeFactors.reweight import GBReweighter
from UserCode.ReweightedFakeFactors.ff_ml_tools import *
from hep_ml.metrics_utils import ks_2samp_weighted

# python scripts/ff_reweight_ml.py --channel=mt --year=2017

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Input channel to determine fake factor weights for', default='mt')
parser.add_argument('--year',help= 'Input year to determine fake factor weights for', default='2017')
parser.add_argument('--output',help= 'Write outputs tp here', default='BDTs')
parser.add_argument("--do_W", action='store_true',help="Get W + jets fake factors. If no do_{W,QCD,ttbar,W_mc} not set will do all.")
parser.add_argument("--do_QCD", action='store_true',help="Get qcd fake factors. If no do_{W,QCD,ttbar,W_mc} not set will do all.")
parser.add_argument("--do_ttbar", action='store_true',help="Get ttbar fake factors. If no do_{W,QCD,ttbar,W_mc} not set will do all.")
parser.add_argument("--batch", action='store_true',help="Batch run fake factors")
parser.add_argument("--no_save", action='store_true',help="Do not save models or overwrite normalisation json")
args = parser.parse_args()


if not args.batch:
  # get files and luminoscity for relevant year
  if args.year == '2018':
    lumi = 58826.8469
    params_file = '/vols/cms/gu18/CrabCMSSW/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/scripts/params_mssm_2018.json'
    input_folder = '/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/'
    if args.channel == "mt": data_files = ['SingleMuonA','SingleMuonB','SingleMuonC','SingleMuonD','TauA','TauB','TauC','TauD']
    elif args.channel == "et": data_files = ['EGammaA','EGammaB','EGammaC','EGammaD','TauA','TauB','TauC','TauD']
    elif args.channel == "tt": data_files = ['TauA','TauB','TauC','TauD']
    ttbar_files = ['TTTo2L2Nu','TTToHadronic','TTToSemiLeptonic']
    wjets_files = ['W1JetsToLNu-LO','W2JetsToLNu-LO','W3JetsToLNu-LO','W4JetsToLNu-LO','WJetsToLNu-LO','EWKWMinus2Jets','EWKWPlus2Jets']
    other_files = ['EWKZ2Jets','T-tW-ext1','T-t','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO','DYJetsToLL-LO','DYJetsToLL_M-10-50-LO','Tbar-tW-ext1',
                   'Tbar-t','WWToLNuQQ','WWTo2L2Nu','WZTo1L3Nu','WZTo2L2Q','WZTo3LNu-ext1','WZTo3LNu','ZZTo2L2Nu-ext1','ZZTo2L2Nu-ext2','ZZTo2L2Q','ZZTo4L-ext','ZZTo4L']
  elif args.year == "2017":
    lumi = 41530.
    params_file = '/vols/cms/gu18/CrabCMSSW/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/scripts/params_mssm_2017.json'
    input_folder = '/vols/cms/dw515/Offline/output/MSSM/mssm_2017_v2/'
    if args.channel == "mt": data_files = ['SingleMuonB','SingleMuonC','SingleMuonD','SingleMuonE','SingleMuonF','TauB','TauC','TauD','TauE','TauF']
    elif args.channel == "et": data_files = ['SingleElectronB','SingleElectronC','SingleElectronD','SingleElectronE','SingleElectronF','TauB','TauC','TauD','TauE','TauF']
    elif args.channel == "tt": data_files = ['TauB','TauC','TauD','TauE','TauF']
    ttbar_files = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic'] 
    wjets_files = ['WJetsToLNu-LO','WJetsToLNu-LO-ext','W1JetsToLNu-LO','W2JetsToLNu-LO','W3JetsToLNu-LO','W4JetsToLNu-LO','EWKWMinus2Jets','EWKWPlus2Jets']
    other_files = ['DYJetsToLL-LO','DYJetsToLL-LO-ext1','DY1JetsToLL-LO','DY1JetsToLL-LO-ext','DY2JetsToLL-LO','DY2JetsToLL-LO-ext','DY3JetsToLL-LO','DY3JetsToLL-LO-ext','DY4JetsToLL-LO',
                   'DYJetsToLL_M-10-50-LO','DYJetsToLL_M-10-50-LO-ext1','T-tW', 'Tbar-tW','Tbar-t','T-t','WWToLNuQQ','WZTo2L2Q','WZTo1L1Nu2Q','WZTo1L3Nu','WZTo3LNu','ZZTo2L2Nu','WWTo2L2Nu',
                   'ZZTo2L2Q','ZZTo4L-ext','ZZTo4L','EWKZ2Jets']
  elif args.year == "2016":
    lumi = 35920.
    params_file = '/vols/cms/gu18/CrabCMSSW/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/scripts/params_mssm_2016.json'
    input_folder = '/vols/cms/dw515/Offline/output/MSSM/mssm_2016_v2/'
    if args.channel == "mt": data_files = ['SingleMuonB','SingleMuonC','SingleMuonD','SingleMuonE','SingleMuonF','SingleMuonG','SingleMuonH','TauB','TauC','TauD','TauE','TauF','TauG','TauH']
    elif args.channel == "et": data_files = ['SingleElectronB','SingleElectronC','SingleElectronD','SingleElectronE','SingleElectronF','SingleElectronG','SingleElectronH','TauB','TauC','TauD',
                                        'TauE','TauF','TauG','TauH']
    elif args.channel == "tt": data_files = ['TauB','TauC','TauD','TauE','TauF','TauG','TauH']
    ttbar_files = ['TT']
    wjets_files = ['WJetsToLNu-LO', 'WJetsToLNu-LO-ext','W1JetsToLNu-LO','W2JetsToLNu-LO','W2JetsToLNu-LO-ext','W3JetsToLNu-LO','W3JetsToLNu-LO-ext','W4JetsToLNu-LO','W4JetsToLNu-LO-ext1',
                   'W4JetsToLNu-LO-ext2', 'EWKWMinus2Jets_WToLNu','EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1','EWKWPlus2Jets_WToLNu-ext2']
    other_files = ['DYJetsToLL-LO-ext1','DYJetsToLL-LO-ext2','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO','DYJetsToLL_M-10-50-LO', 'T-tW', 'Tbar-tW','Tbar-t','T-t','WWTo1L1Nu2Q',
                   'WZJToLLLNu','VVTo2L2Nu','VVTo2L2Nu-ext1','ZZTo2L2Q','ZZTo4L-amcat','WZTo2L2Q','WZTo1L3Nu','WZTo1L1Nu2Q','EWKZ2Jets_ZToLL','EWKZ2Jets_ZToLL-ext1','EWKZ2Jets_ZToLL-ext2']
    
  # read params from json
  with open(params_file) as jsonfile:
    params = json.load(jsonfile)
  
  # get selection for relevant year and channel with alternative working points
  if args.year == "2016": m_lowpt,e_lowpt,t_highpt,t_lowpt_mt,t_lowpt_et = 23,26,120,25,25
  elif args.year == "2017": m_lowpt,e_lowpt,t_highpt,t_lowpt_mt,t_lowpt_et = 25,28,180,32,35
  elif args.year == "2018": m_lowpt,e_lowpt,t_highpt,t_lowpt_mt,t_lowpt_et = 25,33,180,32,35
  if args.channel == "mt":
    baseline = '(deepTauVsJets_vvvloose_2>0.5 && deepTauVsEle_vvloose_2>0.5 && deepTauVsMu_tight_2>0.5 && leptonveto==0 && pt_2>30 && ((trg_mutaucross==1&&pt_2>%(t_lowpt_mt)s&&pt_2<%(t_highpt)s&&fabs(eta_2)<2.1&&pt_1<%(m_lowpt)s)||(trg_singlemuon==1&&pt_1>=%(m_lowpt)s)||(trg_singletau_2==1&&pt_2>=%(t_highpt)s&&fabs(eta_2)<2.1)))' % vars()
  elif args.channel == "et":
    baseline = '(deepTauVsJets_vvvloose_2>0.5 && deepTauVsEle_tight_2>0.5 && deepTauVsMu_vloose_2>0.5 && pt_2>30 && ((trg_etaucross==1&&pt_2>%(t_lowpt_et)s&&pt_2<%(t_highpt)s&&fabs(eta_2)<2.1&&pt_1<%(e_lowpt)s)||(trg_singleelectron==1&&pt_1>=%(e_lowpt)s)||(trg_singletau_2==1&&pt_2>=%(t_highpt)s&&fabs(eta_2)<2.1)))' % vars()
  elif args.channel == "tt":
    baseline = '(deepTauVsJets_vvvloose_1>0.5 && deepTauVsJets_vvvloose_2>0.5 && leptonveto==0 && (trg_doubletau==1 || (pt_1>%(t_highpt)s && trg_singletau_1==1) || (pt_2>%(t_highpt)s && trg_singletau_2==1)) && deepTauVsEle_vvloose_1==1 && deepTauVsEle_vvloose_2==1 && deepTauVsMu_vloose_1==1 && deepTauVsMu_vloose_2==1)' % vars()
  
  # variables and weights needed for training
  if args.channel == "mt":
    X_vars = ["pt_1","pt_2","jet_pt_2","jet_pt_2/pt_2","n_jets","n_deepbjets","tau_decay_mode_2","trg_mutaucross","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"]
    scoring_vars = ['mt_tot','pt_1','pt_2','m_vis','met','eta_2','n_jets','n_deepbjets']
    selection_vars = ["os","mt_1","gen_match_2","deepTauVsJets_medium_2","iso_1"]
  elif args.channel == "et":
    X_vars = ["pt_1","pt_2","jet_pt_2","jet_pt_2/pt_2","n_jets","n_deepbjets","tau_decay_mode_2","trg_etaucross","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"]
    scoring_vars = ['mt_tot','pt_1','pt_2','m_vis','met','eta_2','n_jets','n_deepbjets']
    selection_vars = ["os","mt_1","gen_match_2","deepTauVsJets_medium_2","iso_1"]
  elif args.channel == "tt":
    X_vars = ["pt_1","pt_2","jet_pt_1/pt_1","jet_pt_2/pt_2","jet_pt_1","jet_pt_2","n_jets","n_deepbjets","tau_decay_mode_1","tau_decay_mode_2","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"]
    scoring_vars = ['mt_tot','pt_1','pt_2','m_vis','met','eta_1','eta_2','n_jets','n_deepbjets']
    selection_vars = ["os","deepTauVsJets_vvloose_1","deepTauVsJets_vvloose_2","deepTauVsJets_medium_1","deepTauVsJets_medium_2"]
  
  weights = ["wt","wt_tau_trg_mssm","wt_tau_id_mssm"]
  other_vars = []
  
  # read root files into dataframes, performing initial cuts
  if (args.do_W or args.do_QCD) or not (args.do_ttbar or args.do_W or args.do_QCD):
    print ">> Converting root files into dataframes for data"
    data_df_concat = GetDataframe(baseline,lumi,params,input_folder,data_files,args.channel,args.year,X_vars,weights,selection_vars,scoring_vars,other_vars,data=True)
  
  if (args.do_W or not (args.do_ttbar or args.do_W or args.do_QCD)) and args.channel != "tt":
    print ">> Converting root files into dataframes for W + jets MC"
    wjets_df_concat = GetDataframe(baseline,lumi,params,input_folder,wjets_files+other_files,args.channel,args.year,X_vars,weights,selection_vars,scoring_vars,other_vars)
  
  if (args.do_ttbar or not (args.do_ttbar or args.do_W or args.do_QCD)) and args.channel != "tt":
    print ">> Converting root files into dataframes for ttbar MC"
    ttbar_df_concat = GetDataframe(baseline,lumi,params,input_folder,ttbar_files,args.channel,args.year,X_vars,weights,selection_vars,scoring_vars,other_vars)
  
  print ">> All root files converted to dataframes"

  json_name = '{}/ff_reweight_normalisation.json'.format(args.output)
  if os.path.isfile(json_name):
    with open(json_name) as json_file:
      norm_dict = json.load(json_file)
  else:
    norm_dict = {'et':{'2016':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0},
                       '2017':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0},
                       '2018':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0}},
                 'mt':{'2016':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0},
                       '2017':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0},
                       '2018':{'qcd':0,'qcd_aiso':0,'wjets':0,'wjets_mc':0,'ttbar':0,'qcd_dr_to_ar':0,'wjets_dr_correction':0,'wjets_dr_to_ar':0,'ttbar_dr_to_ar':0}},
                 'tt':{'2016':{'qcd_lead':0,'qcd_sublead':0,'qcd_double':0,'qcd_aiso_lead':0,'qcd_aiso_sublead':0,'qcd_lead_dr_to_ar':0,'qcd_sublead_dr_to_ar':0},
                       '2017':{'qcd_lead':0,'qcd_sublead':0,'qcd_double':0,'qcd_aiso_lead':0,'qcd_aiso_sublead':0,'qcd_lead_dr_to_ar':0,'qcd_sublead_dr_to_ar':0},
                       '2018':{'qcd_lead':0,'qcd_sublead':0,'qcd_double':0,'qcd_aiso_lead':0,'qcd_aiso_sublead':0,'qcd_lead_dr_to_ar':0,'qcd_sublead_dr_to_ar':0}}}


  # perform training and test in each determination region
  if args.channel in ["mt","et"]:
    # get W + jets fake factors
    if args.do_W or not (args.do_ttbar or args.do_W or args.do_QCD):
  
      print ">> Fitting the W + jets weight"
      original = CutAndScale(data_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && n_deepbjets==0 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(data_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && n_deepbjets==0 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/wjets_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      wjets_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['wjets'] = normalisation
      PrintMinMeanMax(original,wjets_reweighter,X_vars,normalisation)
      print ""

      print ">> Fitting the W + jets MC weight"
      original = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && n_deepbjets==0 && (gen_match_2==5 || gen_match_2==6) && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && n_deepbjets==0 && (gen_match_2==5 || gen_match_2==6) && deepTauVsJets_medium_2==1)",1)
      filename = '{}/wjets_mc_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      wjets_mc_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['wjets_mc'] = normalisation
      PrintMinMeanMax(original,wjets_mc_reweighter,X_vars,normalisation)
      print ""

      print ">> Fitting the DR correction for W + jets using MC"
      original = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==0)",norm_dict[args.channel][args.year]['wjets_mc'])
      wjets_mc_ar_weights = wjets_mc_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      original.loc[:,"scale"] = original.loc[:,"scale"].multiply(wjets_mc_ar_weights)
      target = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1>70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/wjets_dr_correction_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      wjets_dr_correction_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[20],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['wjets_dr_correction'] = normalisation
      PrintMinMeanMax(original,wjets_dr_correction_reweighter,X_vars,normalisation)
      print ""

      print ">> Fitting the DR to AR correction for W + jets using MC"
      original = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==0)",norm_dict[args.channel][args.year]['wjets_mc']*norm_dict[args.channel][args.year]['wjets_dr_correction'])
      wjets_mc_ar_weights = wjets_mc_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      wjets_mc_ar_correction_weights = wjets_dr_correction_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      original.loc[:,"scale"] = original.loc[:,"scale"].multiply(wjets_mc_ar_weights).multiply(wjets_mc_ar_correction_weights)
      target = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/wjets_dr_to_ar_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      wjets_dr_to_ar_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[40],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['wjets_dr_to_ar'] = normalisation
      PrintMinMeanMax(original,wjets_dr_to_ar_reweighter,X_vars,normalisation)
      print ""

      print ">> Fitting the W + jets MC weight to compare to ttbar"
      original = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(wjets_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/wjets_mc_comparison_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      wjets_mc_comparison_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['wjets_mc_comparison'] = normalisation
      PrintMinMeanMax(original,wjets_mc_comparison_reweighter,X_vars,normalisation)
 
 
    # get ttbar fake factors
    if args.do_ttbar or not (args.do_ttbar or args.do_W or args.do_QCD):
      print ""
      print ">> Fitting the ttbar weight"
      original = CutAndScale(ttbar_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(ttbar_df_concat,"(iso_1<0.15 && mt_1<70 && os==1 && gen_match_2==6 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/ttbar_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      ttbar_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['ttbar'] = normalisation
      PrintMinMeanMax(original,ttbar_reweighter,X_vars,normalisation)

  # get QCD fake factors
  if args.do_QCD or not (args.do_ttbar or args.do_W or args.do_QCD):
  
    if args.channel in ["mt","et"]:
      # possibly add iso_1>0.05, unsure how to correct for the isolation shift
      print ""
      print ">> Fitting the QCD weight"
      original = CutAndScale(data_df_concat,"(iso_1<0.15 && mt_1<50 && os==0 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(data_df_concat,"(iso_1<0.15 && mt_1<50 && os==0 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/qcd_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['qcd'] = normalisation
      PrintMinMeanMax(original,qcd_reweighter,X_vars,normalisation)
      print ""

      # Problem here, seems to be too low stat for et for algorithm to work for anti-isolated data
      # check why this is the case
      # possibly lower iso_1 bottom cut
      print ">> Fitting the QCD anti-isolated weight"
      original = CutAndScale(data_df_concat,"(iso_1<0.5 && iso_1>0.25 && mt_1<50 && os==0 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(data_df_concat,"(iso_1<0.5 && iso_1>0.25 && mt_1<50 && os==0 && deepTauVsJets_medium_2==1)",1)
      print original
      print target
      filename = '{}/qcd_aiso_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_aiso_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['qcd_aiso'] = normalisation
      PrintMinMeanMax(original,qcd_aiso_reweighter,X_vars,normalisation)
      print ""

      print ">> Fitting the DR to AR correction for QCD using anti-isolated data"
      original = CutAndScale(data_df_concat,"(iso_1<0.5 && iso_1>0.25 && mt_1<70 && os==1 && deepTauVsJets_medium_2==0)",norm_dict[args.channel][args.year]['qcd_aiso'])
      qcd_aiso_ar_weights = qcd_aiso_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      original.loc[:,"scale"] = original.loc[:,"scale"].multiply(qcd_aiso_ar_weights)
      target = CutAndScale(data_df_concat,"(iso_1<0.5 && iso_1>0.25 && mt_1<70 && os==1 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/qcd_dr_to_ar_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_dr_to_ar_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[40],no_save=args.no_save)
      if not args.no_save: norm_dict[args.channel][args.year]['qcd_dr_to_ar'] = normalisation
      PrintMinMeanMax(original,qcd_dr_to_ar_reweighter,X_vars,normalisation)

    elif args.channel == "tt":
 
      print ">> Fitting the QCD leading tau weight"
      original = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_2==1 && deepTauVsJets_medium_1==0)",1)
      target = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_2==1 && deepTauVsJets_medium_1==1)",1)
      filename = '{}/qcd_lead_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_lead_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_lead'] = normalisation
      PrintMinMeanMax(original,qcd_lead_reweighter,X_vars,normalisation)
      print ""
      print ">> Fitting the QCD subleading tau weight"
      original = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_1==1 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_1==1 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/qcd_sublead_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_sublead_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_sublead'] = normalisation
      PrintMinMeanMax(original,qcd_sublead_reweighter,X_vars,normalisation)
      # double tau weight currently not needed
      #print ""
      #print ">> Fitting the QCD double tau weight"
      #original = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_1==0 && deepTauVsJets_medium_2==0)",1)
      #target = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_medium_1==1 && deepTauVsJets_medium_2==1)",1)
      #filename = '{}/qcd_double_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      #qcd_double_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[7],[0.11],[60],no_save=args.no_save)
      #norm_dict[args.channel][args.year]['qcd_double'] = normalisation
      #PrintMinMeanMax(original,qcd_double_reweighter,X_vars,normalisation)
      print ""
      print ">> Fitting the QCD anti-isolated leading tau weight"
      original = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_vvloose_2==0 && deepTauVsJets_medium_1==0)",1)
      target = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_vvloose_2==0 && deepTauVsJets_medium_1==1)",1)
      filename = '{}/qcd_aiso_lead_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_aiso_lead_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_aiso_lead'] = normalisation
      PrintMinMeanMax(original,qcd_aiso_lead_reweighter,X_vars,normalisation)
      print ""
      print ">> Fitting the QCD anti-isolated subleading tau weight"
      original = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_vvloose_1==0 && deepTauVsJets_medium_2==0)",1)
      target = CutAndScale(data_df_concat,"(os==0 && deepTauVsJets_vvloose_1==0 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/qcd_aiso_sublead_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_aiso_sublead_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[80],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_aiso_sublead'] = normalisation
      PrintMinMeanMax(original,qcd_aiso_sublead_reweighter,X_vars,normalisation)
      print ""
      print ">> Fitting the DR to AR correction for QCD leading tau using anti-isolated data"
      original = CutAndScale(data_df_concat,"(os==1 && deepTauVsJets_vvloose_2==0 && deepTauVsJets_medium_1==0)",norm_dict[args.channel][args.year]['qcd_aiso_lead'])
      qcd_aiso_lead_ar_weights = qcd_aiso_lead_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      original.loc[:,"scale"] = original.loc[:,"scale"].multiply(qcd_aiso_lead_ar_weights)
      target = CutAndScale(data_df_concat,"(os==1 && deepTauVsJets_vvloose_2==0 && deepTauVsJets_medium_1==1)",1)
      filename = '{}/qcd_lead_dr_to_ar_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_lead_dr_to_ar_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[40],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_lead_dr_to_ar'] = normalisation
      PrintMinMeanMax(original,qcd_lead_dr_to_ar_reweighter,X_vars,normalisation)
      print ""
      print ">> Fitting the DR to AR correction for QCD subleading tau using anti-isolated data"
      original = CutAndScale(data_df_concat,"(os==1 && deepTauVsJets_vvloose_1==0 && deepTauVsJets_medium_2==0)",norm_dict[args.channel][args.year]['qcd_aiso_sublead'])
      qcd_aiso_sublead_ar_weights = qcd_aiso_sublead_reweighter.predict_weights(original.loc[:,func_to_name(X_vars)],original.loc[:,"scale"],merge_weights=False)
      original.loc[:,"scale"] = original.loc[:,"scale"].multiply(qcd_aiso_sublead_ar_weights)
      target = CutAndScale(data_df_concat,"(os==1 && deepTauVsJets_vvloose_1==0 && deepTauVsJets_medium_2==1)",1)
      filename = '{}/qcd_sublead_dr_to_ar_reweighted_ff_{}_{}.sav'.format(args.output,args.channel,args.year)
      qcd_sublead_dr_to_ar_reweighter, normalisation = FitAndScore(filename,original,target,X_vars,selection_vars,scoring_vars,other_vars,[6],[0.1],[40],no_save=args.no_save)
      norm_dict[args.channel][args.year]['qcd_sublead_dr_to_ar'] = normalisation
      PrintMinMeanMax(original,qcd_sublead_dr_to_ar_reweighter,X_vars,normalisation)

else:
  # batch running doesn't work, as requires too much memory
  cmssw_base = os.getcwd().replace('src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2','')
  cmd = "python scripts/ff_reweight_ml.py --channel={} --year={}".format(args.channel,args.year)
  if args.do_W: cmd += ' --do_W'
  if args.do_QCD: cmd += ' --do_QCD'
  if args.do_W_mc: cmd += ' --do_W_mc'
  if args.do_ttbar: cmd += ' --do_ttbar'
  name = "ff_reweight_{}_{}.sh".format(args.channel,args.year)
  CreateBatchJob(name,cmssw_base,[cmd])
  SubmitBatchJob(name,time=180,memory=24,cores=1)
  

# Write json
if not args.no_save:
  with open(json_name, 'w') as outfile:
    json.dump(norm_dict, outfile)



