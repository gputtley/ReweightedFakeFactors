import uproot
import os
import json
import pandas
import pickle
import argparse
import numpy as np
from UserCode.ReweightedFakeFactors.ff_ml_tools import SelectColumns,to_root

# python scripts/add_ff_reweight_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10 --output_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new --filename=SingleMuonB_mt_2017.root --channel=mt --year=2017 --splitting=100000 --offset=0
# python scripts/add_ff_reweight_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10 --output_location=/vols/cms/gu18/Offline/output/MSSM/reweight_ff_2017 --filename=TauD_tt_2017.root --channel=tt --year=2017 --splitting=100000 --offset=0

parser = argparse.ArgumentParser()
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='./')
parser.add_argument('--filename',help= 'Name of file', default='SingleMuonB_mt_2017.root')
parser.add_argument('--channel',help= 'Name of channel', default='mt')
parser.add_argument('--year',help= 'Name of year', default='2017')
parser.add_argument('--splitting',help= 'Number of events per task', default='100000')
parser.add_argument('--offset',help= 'Offset of job', default='0')
args = parser.parse_args()

columns = {"et":["pt_1","pt_2","jet_pt_2","jet_pt_2/pt_2","n_jets","n_deepbjets","tau_decay_mode_2","trg_etaucross","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"],
           "mt":["pt_1","pt_2","jet_pt_2","jet_pt_2/pt_2","n_jets","n_deepbjets","tau_decay_mode_2","trg_mutaucross","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"],
           "tt":["pt_1","pt_2","jet_pt_1/pt_1","jet_pt_2/pt_2","jet_pt_1","jet_pt_2","n_jets","n_deepbjets","tau_decay_mode_1","tau_decay_mode_2","dR","dphi","pt_1*cosh(eta_1)","pt_2*cosh(eta_2)","met","rho"]}

weights = ["wt","wt_tau_trg_mssm","wt_tau_id_mssm"]

json_name = 'BDTs/ff_reweight_normalisation.json'
with open(json_name) as json_file:
  norm_dict = json.load(json_file)


tree = uproot.open(args.input_location+'/'+args.filename, localsource=uproot.FileSource.defaults)["ntuple"]

k = 0
for small_tree in tree.iterate(entrysteps=int(args.splitting)):
  print k,int(args.offset)
  if k == int(args.offset) or int(args.offset)==-1:
    print k
    df = pandas.DataFrame.from_dict(small_tree)
    for i in weights:
      if i == weights[0]:
        total_weights = df.loc[:,i]
      else:
        total_weights = total_weights.multiply(df.loc[:,i])

    new_df = SelectColumns(df,columns[args.channel])

    if args.channel in ["mt","et"]:

      wjets_mc_reweighter = pickle.load(open("BDTs/wjets_mc_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      wjets_mc_out = wjets_mc_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_wjets_mc_1"] = wjets_mc_out*norm_dict[args.channel][args.year]["wjets_mc"]
  
      wjets_mc_comparison_reweighter = pickle.load(open("BDTs/wjets_mc_comparison_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      wjets_mc_comparison_out = wjets_mc_comparison_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_wjets_mc_comparison_1"] = wjets_mc_comparison_out*norm_dict[args.channel][args.year]["wjets_mc_comparison"]

      wjets_reweighter = pickle.load(open("BDTs/wjets_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      wjets_out = wjets_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_wjets_raw_1"] = wjets_out*norm_dict[args.channel][args.year]["wjets"]

      wjets_dr_to_ar_reweighter = pickle.load(open("BDTs/wjets_dr_to_ar_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      wjets_dr_to_ar_out = wjets_dr_to_ar_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_wjets_dr_to_ar_1"] = wjets_dr_to_ar_out*norm_dict[args.channel][args.year]["wjets_dr_to_ar"]

      wjets_dr_correction_reweighter = pickle.load(open("BDTs/wjets_dr_correction_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      wjets_dr_correction_out = wjets_dr_correction_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_wjets_dr_correction_1"] = wjets_dr_correction_out*norm_dict[args.channel][args.year]["wjets_dr_correction"]


      df.loc[:,"wt_ff_reweight_wjets_dr_1"] = np.multiply(wjets_out,wjets_dr_correction_out)*norm_dict[args.channel][args.year]["wjets"]*norm_dict[args.channel][args.year]["wjets_dr_correction"]
      df.loc[:,"wt_ff_reweight_wjets_1"] = np.multiply(np.multiply(wjets_out,wjets_dr_to_ar_out),wjets_dr_correction_out)*norm_dict[args.channel][args.year]["wjets"]*norm_dict[args.channel][args.year]["wjets_dr_to_ar"]*norm_dict[args.channel][args.year]["wjets_dr_correction"]

      
      qcd_reweighter = pickle.load(open("BDTs/qcd_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_out = qcd_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_raw_1"] = qcd_out*norm_dict[args.channel][args.year]["qcd"]

      qcd_aiso_reweighter = pickle.load(open("BDTs/qcd_aiso_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_aiso_out = qcd_aiso_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_aiso_1"] = qcd_aiso_out*norm_dict[args.channel][args.year]["qcd_aiso"]

      qcd_dr_to_ar_reweighter = pickle.load(open("BDTs/qcd_dr_to_ar_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_dr_to_ar_out = qcd_dr_to_ar_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_dr_to_ar_1"] = qcd_dr_to_ar_out*norm_dict[args.channel][args.year]["qcd_dr_to_ar"]
      
      df.loc[:,"wt_ff_reweight_qcd_1"] = np.multiply(qcd_out,qcd_dr_to_ar_out)*norm_dict[args.channel][args.year]["qcd"]*norm_dict[args.channel][args.year]["qcd_dr_to_ar"]

      ttbar_reweighter = pickle.load(open("BDTs/ttbar_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      ttbar_out = ttbar_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_ttbar_1"] = ttbar_out*norm_dict[args.channel][args.year]["ttbar"]
    
    elif args.channel == "tt":

      qcd_sublead_reweighter = pickle.load(open("BDTs/qcd_sublead_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_sublead_out = qcd_sublead_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_2"] = qcd_sublead_out*norm_dict[args.channel][args.year]["qcd_sublead"]

      qcd_lead_reweighter = pickle.load(open("BDTs/qcd_lead_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_lead_out = qcd_lead_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_1"] = qcd_lead_out*norm_dict[args.channel][args.year]["qcd_lead"]

      qcd_double_reweighter = pickle.load(open("BDTs/qcd_double_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_double_out = qcd_double_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_double"] = qcd_double_out*norm_dict[args.channel][args.year]["qcd_double"]

      qcd_aiso_sublead_reweighter = pickle.load(open("BDTs/qcd_aiso_sublead_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_aiso_sublead_out = qcd_aiso_sublead_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_aiso_2"] = qcd_aiso_sublead_out*norm_dict[args.channel][args.year]["qcd_aiso_sublead"]

      qcd_aiso_lead_reweighter = pickle.load(open("BDTs/qcd_aiso_lead_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_aiso_lead_out = qcd_aiso_lead_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_aiso_1"] = qcd_lead_out*norm_dict[args.channel][args.year]["qcd_aiso_lead"]

      qcd_sublead_dr_to_ar_reweighter = pickle.load(open("BDTs/qcd_sublead_dr_to_ar_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_sublead_dr_to_ar_out = qcd_sublead_dr_to_ar_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_dr_to_ar_2"] = qcd_sublead_dr_to_ar_out*norm_dict[args.channel][args.year]["qcd_sublead_dr_to_ar"]

      qcd_lead_dr_to_ar_reweighter = pickle.load(open("BDTs/qcd_lead_dr_to_ar_reweighted_ff_{}_{}.sav".format(args.channel,args.year), 'rb'))
      qcd_lead_dr_to_ar_out = qcd_lead_dr_to_ar_reweighter.predict_weights(new_df,total_weights)
      df.loc[:,"wt_ff_reweight_qcd_dr_to_ar_1"] = qcd_lead_dr_to_ar_out*norm_dict[args.channel][args.year]["qcd_lead_dr_to_ar"]

    to_root(df, args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"), key='ntuple')
    
    del df, new_df, total_weights, small_tree
    if int(args.offset)!=-1: break
  k += 1
print "Finished processing"
