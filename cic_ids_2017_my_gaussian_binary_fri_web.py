
import numpy as np
import pandas as pd
from my_gaussian_mixture import MyGaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import Normalizer, OrdinalEncoder

cids_fri = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
cids_web = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

# Replacing infinite with nan
cids_fri.replace([np.inf, -np.inf], np.nan, inplace=True)
cids_web.replace([np.inf, -np.inf], np.nan, inplace=True) 

# Dropping all the rows with nan values
cids_fri.dropna(inplace=True)
cids_web.dropna(inplace=True)

categorical_columns = [' Label']
numberic_columns = [' Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std', 'Bwd Packet Length Max',
       ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
       ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
       ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
       ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
       ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
       ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
       ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
       ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
       ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
       ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
       'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
       ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
       ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']

ct = ColumnTransformer(transformers = [('normalize', Normalizer(norm='l2'), numberic_columns),
                                       ("label", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10), categorical_columns)], remainder = 'passthrough')

ct.fit(cids_fri)
cids_fri_transformed = ct.transform(cids_fri)
cids_web_transformed = ct.transform(cids_web)

X_train = cids_fri_transformed[:, 0:78]
y_train = cids_fri_transformed[:, 78]

X_val = cids_web_transformed[:, 0:78]
y_val = cids_web_transformed[:,78]

print(y_val.shape)
clf = MyGaussianMixture(
    n_components=2
)

# Train and evaluate your model
clf.fit(X_train, y_train)

# score = clf.score(X_val_selected, y_val)
# Use the trained classifier to predict the classes of the test set
y_pred = clf.predict(X_val)

print(y_val.shape)
# # Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
