from data_manipulation import *

def ET_XG(a):

    raw_data['userid'] = pd.factorize(raw_data['reviewerID'])[0]
    raw_data['videoid'] = pd.factorize(raw_data['asin'])[0]
    #from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    raw_data['time']=sc.fit_transform(raw_data['reviewTime'].values.reshape(-1,1))
    raw_data['numberuser']=sc.fit_transform(raw_data['no_of_users'].values.reshape(-1,1))
    raw_data['numberprod']=sc.fit_transform(raw_data['no_of_products'].values.reshape(-1,1))
    raw_data['reviewTime'] =  pd.to_datetime(raw_data['reviewTime'], format='%Y-%b-%d:%H:%M:%S.%f')    
    raw_data['weekend']=raw_data['reviewTime'].dt.dayofweek>=5
    raw_data['weekend']=raw_data['weekend'].astype(int)
    First = raw_data.loc[:,['userid','videoid']]
    Second = raw_data.loc[:,['userid','videoid','time']]
    Third = raw_data.loc[:,['userid','videoid','time','numberuser','numberprod']]
    y = raw_data.overall

   # from sklearn.model_selection import train_test_split
    train_1,test_1,y_train,y_test = train_test_split(First,y,test_size=0.3,random_state=2017)
    train_2,test_2,y_train,y_test = train_test_split(Second,y,test_size=0.3,random_state=2017)
    train_3,test_3,y_train,y_test = train_test_split(Third,y,test_size=0.3,random_state=2017)
  # a=ExtraTreesRegressor()
    a.fit(train_3,y_train)
    y3 = a.predict(test_3)
    sc = MinMaxScaler(feature_range=(1,5))
    c = mean_squared_error(y_train,a.predict(train_3)), mean_squared_error(y_test,sc.fit_transform(y3.reshape(-1,1)))
    b = mean_squared_error(y_test,y3)
    print('train MSE is {}, test MSE is {}'.format(c,b))

    c3 = y3>=4
    t = y_test>=4
    print('Recommendation_Accuracy:')
    print(accuracy_score(t,c3))
    c31 = y3<=1
    t1 = y_test<=1
    print('Recommendation_Accuracy:')
    print(accuracy_score(t1,c31))
    y_pred3 = []
    y_test3 = []
    for i in range(y3.shape[0]):
        if y3[i]>=4:
            y_pred3.append(1)
        elif y3[i]<4:
            y_pred3.append(0)

    for j in range(y3.shape[0]):
        if np.array(y_test)[j]>=4:
            y_test3.append(1)
        elif np.array(y_test)[j]<4:
            y_test3.append(0)

    #import itertools
    #import matplotlib.pyplot as plt
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    class_names = ['not recommand','recommand']
    cnf_matrix = confusion_matrix(y_test3,y_pred3)
    tn,fp,fn,tp=confusion_matrix(y_test3,y_pred3).ravel()
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*precision*recall/(precision+recall)
    print("precision: " +str(precision)+"\nrecall: "+ str(recall)+"\n f1_score: "+str(f1_score))        
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='rf')


    plt.show()
    return a


