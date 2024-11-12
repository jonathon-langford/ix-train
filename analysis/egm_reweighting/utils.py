import numpy as np
import pandas as pd

# Utility functions

# Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Taken from ICENET: sigmoid(f/1-f) = exp
def rw_transform_with_logits(logits, absMax=30):
    logits = np.clip(logits, -absMax, absMax)
    return np.exp(logits)


# Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var_name_pretty = {
    "probe_pt" : "Probe $p_T$ (GeV)",
    "probe_eta" : "Probe $\\eta$",
    "fixedGridRhoAll" : "$\\rho$",
    "probe_sieie" : "Probe $\\sigma_{i{\\eta}i\\eta}$",
    "probe_ecalPFClusterIso" : "Probe ECALPFClusterIso",
    "probe_trkSumPtHollowConeDR03" : "Probe $\\mathcal{I}^{hollow}_{tk}({\\Delta}R=0.3)$",
    "probe_hcalPFClusterIso" : "Probe HCALPFClusterIso",
    "probe_pfChargedIsoPFPV" : "Probe PFChargedIso (primary-vertex)",
    "probe_phiWidth" : "Probe $\\phi$-width",
    "probe_trkSumPtSolidConeDR04" : "Probe $\\mathcal{I}^{solid}_{tk}({\\Delta}R=0.4)$",
    "probe_r9" : "Probe $R_{9}$",
    "probe_pfChargedIsoWorstVtx" : "Probe PFChargedIso (worst-vertex)",
    "probe_s4" : "Probe $S_{4}$",
    "probe_etaWidth" : "Probe $\\eta$-width",
    "probe_mvaID" : "Probe mvaID",
    "probe_sieip" : "Probe $\\sigma_{i{\\eta}i\\phi}$",
    "probe_pfChargedIso" : "Probe PFChargedIso",
    "probe_esEffSigmaRR" : "Probe esEffSigmaRR",
    "probe_esEnergyOverRawE" : "Probe esEnergyOverRawE",
}
