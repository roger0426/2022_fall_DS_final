import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from NN_model import NN_sklearn_wrapper
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main():
    one_hot_cols = ['ind_nuevo', 'indrel', 'indext', 'ind_actividad_cliente',
                    'sexo_female', 'sexo_male', 'canal_entrada_007', 'canal_entrada_013',
                    'canal_entrada_KAA', 'canal_entrada_KAB', 'canal_entrada_KAE', 'canal_entrada_KAF',
                    'canal_entrada_KAG', 'canal_entrada_KAH', 'canal_entrada_KAI', 'canal_entrada_KAJ',
                    'canal_entrada_KAP', 'canal_entrada_KAQ', 'canal_entrada_KAR', 'canal_entrada_KAS',
                    'canal_entrada_KAT', 'canal_entrada_KAW', 'canal_entrada_KAY', 'canal_entrada_KAZ',
                    'canal_entrada_KBZ', 'canal_entrada_KCC', 'canal_entrada_KCH', 'canal_entrada_KCI',
                    'canal_entrada_KEY', 'canal_entrada_KFA', 'canal_entrada_KFC', 'canal_entrada_KFD',
                    'canal_entrada_KHC', 'canal_entrada_KHD', 'canal_entrada_KHE', 'canal_entrada_KHF',
                    'canal_entrada_KHK', 'canal_entrada_KHL', 'canal_entrada_KHM', 'canal_entrada_KHN',
                    'canal_entrada_KHQ', 'canal_entrada_RED', 'canal_entrada__others', 'nomprov_ALAVA',
                    'nomprov_ALBACETE', 'nomprov_ALICANTE', 'nomprov_ALMERIA', 'nomprov_ASTURIAS',
                    'nomprov_AVILA', 'nomprov_BADAJOZ', 'nomprov_BALEARS, ILLES', 'nomprov_BARCELONA',
                    'nomprov_BIZKAIA', 'nomprov_BURGOS', 'nomprov_CACERES', 'nomprov_CADIZ', 'nomprov_CANTABRIA',
                    'nomprov_CASTELLON', 'nomprov_CEUTA', 'nomprov_CIUDAD REAL', 'nomprov_CORDOBA',
                    'nomprov_CORUÑA, A', 'nomprov_CUENCA', 'nomprov_GIPUZKOA', 'nomprov_GIRONA', 'nomprov_GRANADA',
                    'nomprov_GUADALAJARA', 'nomprov_HUELVA', 'nomprov_HUESCA', 'nomprov_JAEN', 'nomprov_LEON',
                    'nomprov_LERIDA', 'nomprov_LUGO', 'nomprov_MADRID', 'nomprov_MALAGA', 'nomprov_MELILLA',
                    'nomprov_MURCIA', 'nomprov_NAVARRA', 'nomprov_OURENSE', 'nomprov_PALENCIA', 'nomprov_PALMAS, LAS',
                    'nomprov_PONTEVEDRA', 'nomprov_RIOJA, LA', 'nomprov_SALAMANCA', 'nomprov_SANTA CRUZ DE TENERIFE',
                    'nomprov_SEGOVIA', 'nomprov_SEVILLA', 'nomprov_SORIA', 'nomprov_TARRAGONA', 'nomprov_TERUEL',
                    'nomprov_TOLEDO', 'nomprov_VALENCIA', 'nomprov_VALLADOLID', 'nomprov_ZAMORA', 'nomprov_ZARAGOZA',
                    'segmento_01 - TOP', 'segmento_02 - PARTICULARES', 'segmento_03 - UNIVERSITARIO']

    numerical_cols = ['age', 'antiguedad', 'renta']

    history = ['ind_ahor_fin_ult1_history', 'ind_aval_fin_ult1_history', 'ind_cco_fin_ult1_history',
               'ind_cder_fin_ult1_history', 'ind_cno_fin_ult1_history', 'ind_ctju_fin_ult1_history',
               'ind_ctma_fin_ult1_history', 'ind_ctop_fin_ult1_history', 'ind_ctpp_fin_ult1_history',
               'ind_deco_fin_ult1_history', 'ind_dela_fin_ult1_history', 'ind_deme_fin_ult1_history',
               'ind_ecue_fin_ult1_history', 'ind_hip_fin_ult1_history', 'ind_fond_fin_ult1_history',
               'ind_plan_fin_ult1_history', 'ind_pres_fin_ult1_history', 'ind_reca_fin_ult1_history',
               'ind_tjcr_fin_ult1_history', 'ind_valo_fin_ult1_history', 'ind_viv_fin_ult1_history',
               'ind_nomina_ult1_history', 'ind_nom_pens_ult1_history', 'ind_recibo_ult1_history']

    '''
    product = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
            'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
            'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
            'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
            'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
            'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
    '''
    target_col = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                  'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                  'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                  'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                  'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                  'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    # month3 = ['ind_ahor_fin_ult1_3month', 'ind_aval_fin_ult1_3month', 'ind_cco_fin_ult1_3month',
    #           'ind_cder_fin_ult1_3month', 'ind_cno_fin_ult1_3month', 'ind_ctju_fin_ult1_3month',
    #           'ind_ctma_fin_ult1_3month', 'ind_ctop_fin_ult1_3month', 'ind_ctpp_fin_ult1_3month',
    #           'ind_deco_fin_ult1_3month', 'ind_deme_fin_ult1_3month', 'ind_dela_fin_ult1_3month',
    #           'ind_ecue_fin_ult1_3month', 'ind_fond_fin_ult1_3month', 'ind_hip_fin_ult1_3month',
    #           'ind_plan_fin_ult1_3month', 'ind_pres_fin_ult1_3month', 'ind_reca_fin_ult1_3month',
    #           'ind_tjcr_fin_ult1_3month', 'ind_valo_fin_ult1_3month', 'ind_viv_fin_ult1_3month',
    #           'ind_nomina_ult1_3month', 'ind_nom_pens_ult1_3month', 'ind_recibo_ult1_3month']

    path = './debug.pkl'
    # path = '../data/train_preprocessed_v4.pkl'
    method = 'Nerual Network'
    data = pd.read_pickle(path)

    scaled_features = StandardScaler().fit_transform(
        data[numerical_cols].values)
    x = np.concatenate(
        (data[one_hot_cols].values, scaled_features), axis=1)

    for target in target_col:
        print(target)
        X = np.concatenate(
            (x, np.expand_dims(data[target + '_history'].values, axis=1)), axis=1)

        y = data[target + '_3month'].values

        best_model = f"./result/{method}/{target}_model.pkl"
        with open(best_model, 'rb') as fp:
            model: NN_sklearn_wrapper = pickle.load(fp)
        vecs = model.get_latent(X)
        print(f"latent shape={vecs.shape}")  # (1361948, 256)
        decomp = PCA(n_components=2)
        vecs = decomp.fit_transform(vecs)
        os.makedirs(os.path.join('./plot', target), exist_ok=True)
        for hue in ['ind_nuevo', 'indrel', 'indext', 'ind_actividad_cliente', 'age', 'antiguedad', 'renta']:
            print(hue, end=', ')
            plt.figure()
            sns.scatterplot(
                data=data,
                x=vecs[:, 0],
                y=vecs[:, 1],
                hue=hue
            ).set_title(f"Product: {target}, Label: {hue}")
            plt.savefig(os.path.join('./plot', target, hue))
            plt.close()


if __name__ == '__main__':
    main()
