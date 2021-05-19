import ROOT
import os
from UserCode.ReweightedFakeFactors.ff_ml_tools import CreateBatchJob,SubmitBatchJob
import argparse

#python scripts/batch_add_ff_reweight_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10 --output_location=/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10_new

parser = argparse.ArgumentParser()
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/mssm_2017_v10')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/reweight_ff_2017')
parser.add_argument("--hadd", action='store_true',help="Hadd files after adding weights")
args = parser.parse_args()

loc = args.input_location
newloc = args.output_location

splitting = 100000 

cmssw_base = os.getcwd().replace('src/UserCode/ReweightedFakeFactors','')

for file_name in os.listdir(loc):
  if ".root" in file_name:
    if "_et_" in file_name:
      channel = "et"
    elif "_mt_" in file_name:
      channel = "mt"
    elif "_tt_" in file_name:
      channel = "tt"

    if "_2016" in file_name:
      year = "2016"
    elif "_2017" in file_name:
      year = "2017"
    elif "_2018" in file_name:
      year = "2018"


    if not args.hadd:
      f = ROOT.TFile(loc+'/'+file_name,"READ")
      t = f.Get("ntuple")
      ent = t.GetEntries()
      splits = ((ent - (ent%splitting))/splitting) + 1

      for i in range(0,splits):
        cmd = "python scripts/add_ff_reweight_to_trees.py --input_location=%(loc)s --output_location=%(newloc)s --filename=%(file_name)s --channel=%(channel)s --year=%(year)s --splitting=%(splitting)i --offset=%(i)i" % vars()
        CreateBatchJob('jobs/'+file_name.replace('.root','_'+str(i)+'.sh'),cmssw_base,[cmd])
        SubmitBatchJob('jobs/'+file_name.replace('.root','_'+str(i)+'.sh'),time=180,memory=24,cores=1)

     else:
       os.system("hadd -f {}/{} {}/{}".format(newloc,filename,newloc,filename.replace(".root","_*")))
       os.system("rm {}/{}".format(newloc,filename.replace(".root","_*")))


     
