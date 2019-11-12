from mongoengine import connect
from pycoshark.mongomodels import Issue
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, train_test_split

from icb.approaches.antoniol2008 import Antoniol2008
from icb.approaches.chawla2015 import Chawla2015
from icb.approaches.kallis2019 import Kallis2019
from icb.approaches.limsettho2014 import Limsettho2014
from icb.approaches.otoom2019 import Otoom2019
from icb.approaches.pandey2018 import Pandey2018
from icb.approaches.pingclasai2013 import Pingclasai2013
from icb.approaches.qin2018 import Qin2018
from icb.approaches.terdchanakul2017 import Terdchanakul2017
from icb.utils import create_data_frame_for_issue_data


if __name__ == "__main__":
    swe = {'host': 'XXXXXXXXXX',
           'port': 27017,
           'db': 'smartshark',
           'username': 'XXXXXXXXX',
           'password': 'XXXXXXXXXXX',
           'authentication_source': 'XXXXXXXX',
           'connect': False}
    connect(**swe)

    # Get issues
    issues = Issue.objects.filter(issue_system_id='5b6403f596444971f1cc2d78', issue_type_verified__exists=True).all()[0:200]

    df = create_data_frame_for_issue_data(issues)
    df = df.dropna()

    # Train/Test setup
    for approach in [Antoniol2008(LogisticRegression(solver='lbfgs')), Chawla2015(), Kallis2019(), Limsettho2014(LogisticRegression(solver='lbfgs')),
                     Otoom2019(LogisticRegression(solver='lbfgs')), Pandey2018(LogisticRegression(solver='lbfgs')), Pingclasai2013(LogisticRegression(solver='lbfgs')),
                     Qin2018(), Terdchanakul2017(LogisticRegression(solver='lbfgs'), '/home/ftrauts/Arbeit/projects/ngweight/bin/default/ngweight', feature_selector=SelectKBest(chi2, k=10))]:
           print("Using approach %s ..." % approach.__class__.__name__)
           X_train, X_test, y_train, y_test = train_test_split(approach.filter(df), df.classification, test_size=0.2)
           approach.fit(X_train, y_train)
           predictions = approach.predict(X_test)
           print(accuracy_score(y_test, predictions))

    # CV-Setup
    for approach in [Antoniol2008(LogisticRegression(solver='lbfgs')), Chawla2015(), Kallis2019(), Limsettho2014(LogisticRegression(solver='lbfgs')),
                     Otoom2019(LogisticRegression(solver='lbfgs')), Pandey2018(LogisticRegression(solver='lbfgs')), Pingclasai2013(LogisticRegression(solver='lbfgs')),
                     Qin2018(), Terdchanakul2017(LogisticRegression(solver='lbfgs'), '/home/ftrauts/Arbeit/projects/ngweight/bin/default/ngweight', feature_selector=SelectKBest(chi2, k=10))]:
           print("Using approach %s ..." % approach.__class__.__name__)
           scores = cross_validate(approach, approach.filter(df), df.classification, cv=5, scoring=['f1', 'accuracy'])
           print(scores)




