# read data from csv

# understand the dataset
## shape of data
## type of data

# clean dataset
## make all column names lower case for easy understanding
## make all column string values lower case to easy filtering
## remove space and replace with underscore in column name to make it consistant while filtering or selection
## remove space and replace with underscore in column string values to make it consistant while filtering

# explore dataset
## understand holistic view how data is distributed for target variable.
## if required convert to logrithmic form

# Data Validation - Divide source shuffled dataset into 3 data 
## train data with 60% dataset
## validate data with 20 % dataset
## test data with 20%  dataset
# Divide data into feature and target 

# understanding bias vairble and weights for each feaures

# Training a linear regression model


```python
XTX=np.dot(X.T,X)
XTX_inv=np.linalg.inv(XTX)
XTX_inv.dot(X.T).dot(y) # w_new

ones=np.ones(X.shape[0])
X=np.column_stack((ones,X))
# adding bias term to feature matrix

w_full=XTX_inv.dot(X.T).dot(y_train)
w0=w_full[0]
w=w_full[1:]

return w0,w
```

X - feature matrix i.e x_train=df['Numerical features'].values

y - target variable

predictions= w0 + X.dot(w) 


# Verify prediction using plots

```
sns.histplot(predictions,bins=50,alpha=0.5,color='red') # predicted
sns.histplot(y_train,bins=50,alpha=0.5,color='green') # target

```

# RMSE

```
ef rmse(y,y_pred):

    se=(y - y_pred)**2
    mse=se.mean()
    return np.sqrt(mse)

rmse(y_train,predictions)
```
Lower the value better the results.
If the value is on high side evaluating a model, whether by adding new columns can prediction be improved.


