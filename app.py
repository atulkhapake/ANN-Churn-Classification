#!/usr/bin/env python
# coding: utf-8

# In[18]:


## load the trained model,scaler pickle onehot

model=tf.keras.models.load_model('model.h5')
with open('geo_encoder.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)
with open('gender_encoding.pkl', 'rb') as file:
    loaded_gender_encoder = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


# In[19]:


## Stremlit App

st.title('CUSTOMER CHURN PREDICTION')
#input
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender',loaded_gender_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit_score')
estimated_salary=st.number_input('Estimated_salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


# In[20]:


#Input
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[loaded_gender_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


# In[21]:


geo_encoded = label_encoder_geo.transform(np.array([[geography]]))


# In[22]:


geo_encoded_df=pd.DataFrame(geo_encoded,
columns=label_encoder_geo.get_feature_names_out(['Geography']))


# In[23]:


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


# In[24]:


input_data


# In[25]:


input_data_scaled=scaler.transform(input_data)


# In[26]:


input_data_scaled


# In[27]:


prediction=model.predict(input_data_scaled)


# In[28]:


prediction_proba=prediction[0][0]


# In[29]:


prediction_proba


# In[30]:


if prediction_proba>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")

