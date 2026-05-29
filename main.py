import pandas as pd
import numpy as np

df = pd.read_csv("Bengaluru_House_Data.csv")

#Handling null values
df= df.dropna(subset = 'location')
df['size'] = df['size'].fillna(df['size'].mode()[0])
df['bath'] = df['bath'].fillna(df['bath'].mode()[0])
df['balcony'] = df['balcony'].fillna(df['balcony'].mode()[0])
df['society'] = df['society'].fillna(df['society'].mode()[0])

#print(df.isnull().sum())


#Data Cleaning

# Making function to split total_sqft column extracting the num value
# [a-b, 6Sq. Meter,5Perch,1Acres]

import re
def avg(x):
    if isinstance(x, str):
        x = x.strip()
        # Case 1: range
        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        # Extract number
        nums = re.findall(r'\d+\.?\d*', x)
        if not nums:
            return None
        num = float(nums[0])

        # Unit conversion
        if 'Sq. Meter' in x:
            return num * 10.7639
        elif 'Sq. Yard' in x:
            return num * 9
        elif 'Acre' in x:
            return num * 43560
        elif 'Perch' in x:
            return num * 272.25
        else:
            return num  # already in sqft

    return x

df['total_sqft'] = df['total_sqft'].apply(avg)


#Convert locations having value counts less
#than 10 as 'others' in location column

count = df["location"].value_counts()
def other(x):
  if count[x]<20:
    return 'others'
  else:
    return x


df['location'] = df['location'].apply(other)

#Removing outliers from bath
q1 = df['bath'].quantile(0.25)
q3 = df['bath'].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df['bath'] >= lower) & (df['bath'] <= upper)]



#Feature engineering

# Feature extraction from the column size making new bhk column
# by extracting number from the string
import re
def get(x):
  num = re.findall(r'\d+',x)
  return int(num[0])
df['bhk'] = df['size'].apply(get)

#Removing outliers from bhk
q1 = df['bhk'].quantile(0.25)
q3 = df['bhk'].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df['bhk'] >= lower) & (df['bhk'] <= upper)]

#Removing not practical rows where bath are more than bhk
df = df[df['bath'] <= df['bhk'] + 1]
#Removing outliers from total_sqft
q1 = df['total_sqft'].quantile(0.25)
q3 = df['total_sqft'].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df['total_sqft'] >= lower) & (df['total_sqft'] <= upper)]

#drop the column size
df = df.drop('size', axis=1)
#drop the society size

df = df.drop('society', axis=1)

# Included all the dates columns in one category "Soon to be vacated "
#df['availability'].value_counts()

def soon_vacated(x):
  if isinstance(x,str):
    if "-" in x:
      return "Soon to be Vacated"
    else:
      return x
df['availability'] = df['availability'].apply(soon_vacated)



from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)



#ENcoding string columns

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle

dict_encoders = {}

categorical_columns = [
    'area_type',
    'availability',
]
for cols in categorical_columns:
  le = LabelEncoder()
  df[cols] = le.fit_transform(df[cols])
  dict_encoders[cols] = le

encoder = OneHotEncoder(sparse_output =False)
encoded = encoder.fit_transform(df[['location']])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['location']),
    index=df.index
)
df = df.drop('location', axis=1)
df = pd.concat([df, encoded_df], axis=1)
dict_encoders['location'] = encoder

with open('encoders.pkl', "wb") as f:
    pickle.dump(dict_encoders, f)



#display(df['total_sqft'].value_counts())
#Machine Learning Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)
from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=20,
    min_samples_split=5
)

RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Saving our LR_model variable
with open("RF_model.pkl","wb") as f:
  pickle.dump(RF_model, f)






#Model Deployment
import streamlit as st

#load trained model
with open("RF_model.pkl","rb") as f:
  model = pickle.load(f)

#load the dictionary of label encoder
with open ("encoders.pkl","rb") as f:
  encoders = pickle.load(f)

st.title("BENGALURU HOUSE PRICE PREDICTION")

area_type_le = encoders['area_type']
Area = st.selectbox('Area Type',area_type_le.inverse_transform(df['area_type'].unique()))
total_sqft = st.slider('Total_Sqft',300.0,10000.0,1200.0)
bath = st.selectbox("Bath",[1,2,3,4,5,6,7,8,9])
availability_le = encoders['availability']
availability = st.selectbox('Availability',availability_le.inverse_transform(df['availability'].unique()))
balcony = st.selectbox("Balcony",[0,1,2,3])
location_le = encoders['location']
location = st.selectbox('Location',location_le.categories_[0])
bhk = st.selectbox('BHK',[1,2,3,4,5,6,7,8,9,10])
if st.button('Predict Price'):

    # Encode label encoded columns
    encoded_area_type = area_type_le.transform([Area])[0]
    encoded_availability = availability_le.transform([availability])[0]

    # One hot encode location
    location_encoded = location_le.transform([[location]])

    # Create base dataframe
    input_data = pd.DataFrame([[
        encoded_area_type,
        total_sqft,
        bath,
        encoded_availability,
        balcony,
        bhk
    ]],
    columns=[
        'area_type',
        'total_sqft',
        'bath',
        'availability',
        'balcony',
        'bhk'
    ])

    # Add location columns
    location_df = pd.DataFrame(
        location_encoded,
        columns=location_le.get_feature_names_out(['location'])
    )

    # Combine both
    input_data = pd.concat([input_data, location_df], axis=1)

    # Match training column order
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted House Price: ₹ {prediction:,.2f} Lakhs')


