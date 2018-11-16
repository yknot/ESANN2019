# Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class RF_generator():
    def __init__(self, ds):
        """ Data generator using multiple imputations with random forest
            Input:
              ds: AutoML object containing data
        """
        # List of Random Forests
        self.models = []

        # Random forest from sklearn
        self.regressor = RandomForestRegressor
        self.classifier = RandomForestClassifier

        # pandas dataset
        self.ds = ds

        # Generated DataFrame
        self.gen_data = self.ds.copy()

    def fit(self):
        """
            Fit one random forest for each column, given the others
        """

        #print('New one')
        #te = list(data.columns)
        for c in self.ds.columns:
            #print(te[i])
            # May bug with duplicate names in columns
            y = self.ds[c]
            X = self.ds.drop(c, axis=1)

            # Regressor or classifier
            model = self.regressor(n_estimators=5)
            #else:
            #    model = self.classifier(n_estimators=5)
            model.fit(X, y)
            self.models.append(model)

    def generate(self):
        """
            :return: Generated data
            :rtype: pd.DataFrame
        """
        predicted_matrix = np.zeros(self.ds.shape)
        residual_matrix = np.zeros(self.ds.shape)
        for x in list(self.ds.index.values):
            for i, y in enumerate(list(self.ds.columns.values)):
                row = self.ds.loc[[x]].drop(y, axis=1)
                predicted_matrix[x, i] = self.models[i].predict(row)
                residual_matrix[x, i] = (
                    predicted_matrix[x, i] - self.ds.loc[x, y])**2
        var_vector = np.mean(residual_matrix, axis=0)
        for i in range(predicted_matrix.shape[0]):
            row = predicted_matrix[i, :]
            for j, y in enumerate(list(self.ds.columns.values)):
                self.gen_data.at[i, y] = row[j] + np.random.normal(
                    loc=0, scale=np.sqrt(var_vector[j]))

        return self.gen_data

    def generate_main(self, filename):
        """ Generate synthetic data and returns numpy matrix
        """
        x = self.generate()
        gen = pd.DataFrame(x, columns=self.ds.columns)
        gen.to_csv(filename, index=False)


data = pd.read_csv('../mimic_train_sdv.csv')
rf = RF_generator(data)
rf.fit()
for i in range(10):
    rf.generate_main(f'mimic_anm_synthetic_{i}.csv')
