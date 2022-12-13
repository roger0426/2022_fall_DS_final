import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from NN_model import NN_sklearn_wrapper
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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

    os.makedirs(f'./result/{method}', exist_ok=True)

    with open(f"./result/{method}/result.log", "w") as fp:

        scaled_features = StandardScaler().fit_transform(
            data[numerical_cols].values)
        x = np.concatenate(
            (data[one_hot_cols].values, scaled_features), axis=1)

        for target in target_col:

            print(target)
            fp.write(f"{target}\n")

            X = np.concatenate(
                (x, np.expand_dims(data[target + '_history'].values, axis=1)), axis=1)

            y = data[target + '_3month'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            best_model = f"./result/{method}/{target}_model.pkl"
            model = NN_sklearn_wrapper(in_features=X_train.shape[1])
            model.fit(X_train, y_train, dev_X=X_test, dev_y=y_test)

            # 計算準確率
            score = model.score(X_train, y_train)
            print('train roc_auc_score: ', score)
            fp.write(f"train roc_auc_score: {score}\n")
            score = model.score(X_test, y_test)
            print('test roc_auc_score: ', score)
            fp.write(f"test roc_auc_score: {score}\n")

            # get confusion matrix
            probs = model.predict_proba(X_test)
            pred = np.argmax(probs, 1)
            confusion_matrix = pd.crosstab(y_test, pred,
                                           rownames=['Actual'],
                                           colnames=['Predicted'])
            plot = sns.heatmap(confusion_matrix, linewidth=.5,
                               annot=True, fmt=',.0f')
            plot.set_title(f'{method} % {target}')
            fig = plot.get_figure()
            fig.savefig(f'./result/{method}/{target}_confmat.png')
            plt.clf()

            # keep probabilities for the positive outcome only
            probs = probs[:, 1]
            # calculate scores
            # auc = roc_auc_score(y_test, probs)
            # calculate roc curves
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{method} % {target} % ROC')
            plt.savefig(f'./result/{method}/{target}_roc.png')
            plt.clf()
            # save the model
            model.save_model(best_model)


if __name__ == '__main__':
    main()
