import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import kneighbors_graph



class TaskBin():
    def __init__(self,path):
        self.path = path 
        self.df=pd.read_excel(path)
        self.smiles=self.df["SMILES"].values

# trasfer smiles into fp
    def featurize(self,featurization_method):


        fingerprints=[]
        
        mols=[Chem.MolFromSmiles(smile) for smile in self.smiles]
        for i in range(len(mols)):
            if featurization_method=="rdkit":
                generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096,countSimulation=False)
            if featurization_method=="atom_pair":
                generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096,countSimulation=False)
            if featurization_method=="top_torsion":
                generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=4096,countSimulation=False)
            if featurization_method=="morgan":
                generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=4096,countSimulation=False)
            #fingerprint=generator.GetFingerprintAsNumPy(mols[i])
            if featurization_method=="avalon":
                fp = pyAvalonTools.GetAvalonFP(mols[i], nBits=4096)
                fingerprints.append(fp)
                continue
            fingerprint=generator.GetFingerprint(mols[i])

            fingerprints.append(fingerprint)
        
        return fingerprints



    def cluster(self,molecules_featurized):
        connectivity = kneighbors_graph(molecules_featurized, n_neighbors=1, include_self=False)
        clusterable_embedding = umap.UMAP(
        n_neighbors=1600,
        min_dist=0.0,
        n_components=2,
        random_state=8,
        ).fit_transform(molecules_featurized)


        #plt.figure(figsize=(24, 16)) 
        clustering = AgglomerativeClustering(n_clusters=15,linkage="average",connectivity=connectivity).fit(clusterable_embedding)
        labels=clustering.labels_
            
        return labels,clusterable_embedding





    def assign_tasks(self):
        df_copy=self.df.copy()
        cmap = plt.get_cmap('tab20', 15)
        molecules_featurized=self.featurize("top_torsion")
        clusters,clusterable_embedding=self.cluster(molecules_featurized)
        df_copy["Task"]=clusters
        return df_copy

