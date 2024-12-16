# from .utils import (compare_binary, compare_gaussian, compare_binary_group,
#                      compare_both_true, compare_rate)
import numpy as np

def compare_binary(feat1,feat2):
    return int(feat1==feat2)


def compare_both_true(feat1,feat2):
    return int(feat1 and feat2)


def compare_gaussian(feat_ref,feat):
    std = abs(feat_ref)*0.05
    return np.exp(-0.5*((feat_ref-feat)/std)**2)


def compare_binary_group(feat1_list,feat2_list):
    if feat1_list == [] and feat2_list == []:
        return 1
    
    value = 0
    for feat1 in feat1_list:
        if feat1 in feat2_list:
            value = 1
            break
    return value


def compare_rate(feat1_dict:dict,feat2_dict:dict):
    size = (sum(feat1_dict.values()) + sum(feat2_dict.values())) / 2
    matching = 0
    for key in feat1_dict.keys():
        if key in feat2_dict:
            matching += min(feat1_dict[key],feat2_dict[key])
    return matching/size

######################################
########## COMPARING NODES ###########
######################################

WEIGHTING_SCHEME = {'atomic_nb':1,
                    'nb_implicit_h':1,
                    'formal_charge':1,
                    'partial_charge':1,
                    'degree':1,
                    # 'position':1,
                    'bond_order':1,
                    # 'pharmacophoric':1,
                    'is_donor':3,
                    'is_acceptor':3,
                    'is_hydrophobic':3,
                    'is_aromatic':3}

WEIGHTING_SCHEME_RINGS = {'atomic_nb':1,
                    'nb_implicit_h':1,
                    'formal_charge':1,
                    'partial_charge':1,
                    'degree':1,
                    # 'position':1,
                    'double':3,
                    'bond_order':1,
                    # 'pharmacophoric':1,
                    'is_donor':3,
                    'is_acceptor':3,
                    'is_hydrophobic':3,
                    'is_aromatic':0}

COMPARE_METHOD = {'atomic_nb':compare_binary,
                    'nb_implicit_h':compare_binary,
                    'formal_charge':compare_binary,
                    'partial_charge':compare_gaussian,
                    'degree':compare_gaussian,
                    # 'position':1,
                    'bond_order':compare_binary,
                    # 'pharmacophoric':compare_binary_group,
                    'is_donor':compare_binary,
                    'is_acceptor':compare_binary,
                    'is_hydrophobic':compare_binary,
                    'is_aromatic':compare_binary}

COMPARE_METHOD_RINGS = {'atomic_nb':compare_rate,
                    'nb_implicit_h':compare_binary,
                    'formal_charge':compare_binary,
                    'partial_charge':compare_gaussian,
                    'degree':compare_gaussian,
                    # 'position':1,
                    'bond_order':compare_binary,
                    'double':compare_binary,
                    # 'pharmacophoric':compare_binary_group,
                    'is_donor':compare_binary,
                    'is_acceptor':compare_binary,
                    'is_hydrophobic':compare_binary,
                    'is_aromatic':compare_binary}


######################################
######## BUILDING CONFLICT G #########
######################################

# CRITICAL_FEATURES = ['atomic_nb','pharmacophoric']
CRITICAL_FEATURES = ['atomic_nb','is_donor','is_acceptor',
                     'is_hydrophobic','is_aromatic']
DISTANCE_THRESHOLD = 0.1  # this is rate

######################################
######## FEATURE DEFINITION ##########
######################################
# """ Definitions for 2D Pharmacophores from:
#   Gobbi and Poppinger, Biotech. Bioeng. _61_ 47-54 (1998)

# """
# taken from: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Pharm2D/Gobbi_Pharm2D.py
# TAKE A LOOK AT RDKIT LICENSE 

fdef_string = """
DefineFeature Hydrophobic [$([C;H2,H1](!=*)[C;H2,H1][C;H2,H1][$([C;H1,H2,H3]);!$(C=*)]),$(C([C;H2,H3])([C;H2,H3])[C;H2,H3])]
  Family LH
  Weights 1.0
EndFeature
DefineFeature Donor [$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]
  Family HD
  Weights 1.0
EndFeature
DefineFeature Acceptor [$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]
  Family HA
  Weights 1.0
EndFeature
DefineFeature AromaticAttachment [$([a;D3](@*)(@*)*)]
  Family AR
  Weights 1.0
EndFeature
DefineFeature AliphaticAttachment [$([A;D3](@*)(@*)*)]
  Family RR
  Weights 1.0
EndFeature
DefineFeature UnusualAtom [!#1;!#6;!#7;!#8;!#9;!#16;!#17;!#35;!#53]
  Family X
  Weights 1.0
EndFeature
DefineFeature BasicGroup [$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([N,n;X2;+0])]
  Family BG
  Weights 1.0
EndFeature
DefineFeature AcidicGroup [$([C,S](=[O,S,P])-[O;H1])]
  Family AG
  Weights 1.0
EndFeature
"""

######################################
########### FEATURE NAMES ############
######################################
FT_NAMES = {
    'Gobbi':{
        'donor': 'Donor',
        'acceptor': 'Acceptor',
        'hydrophobic': 'Hydrophobic',
        'aromatic': ['AromaticAttachment']
    },
    'base':{
        'donor': 'SingleAtomDonor',
        'acceptor': 'SingleAtomAcceptor',
        'hydrophobic': 'Hphobe',
        'aromatic': ['Arom4','Arom5','Arom6','Arom7','Arom8']
    }
}