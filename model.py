import pandas as pd 
import numpy  as np
from sklearn.mixture import GMM

def fit_gmm(location_data):
    models = []
    bic = []
    for n in range(1, 8):
        gmm = GMM(n, covariance_type='full')
        gmm.fit(location_data)
        models.append(gmm)
        bic.append(gmm.bic())
    return models[np.argmin(bic)]


def score_point(gmm, long, lat):
    x = np.array([long, lat])
    s = gmm.score_samples(x)
    return s

class TravelPatternDetector(object):
    def __init__(self, num_slots=48, theta=0):
        self.bounds = []
        num_minutes = 2400 * 60
        num_minutes_per_window = num_minutes / num_slots
        for start in range(0, num_minutes + 1, num_minutes_per_window):
            end = start + num_minutes_per_window
            self.bounds.append((start, end))
        self.n_blocks = len(self.bounds)
        self.gmms = []
        self.theta = theta
    
    def _fit_df(self, df):
        for start, end in self.bounds:
            current = df.loc[df['time'] >= start & df['time'] < end, ['long', 'lat']]
            location = current.values
            gmm = fit_gmm(location)
            self.gmms.append(gmm)

    def _fit_arr(self, X):
        time = X[:,0]
        for start, end in self.bounds:
            current = X[time >= start & time < end,:]
            location = current[:, [1, 2]]
            gmm = fit_gmm(location)
            self.gmms.append(gmm)

    def fit(self, X):
        if type(X) == np.ndarray:
            self._fit_arr(X)
        else:
            self._fit_df(X)
    
    def score(self, time, long, lat):
        gmm = None
        for i, (start, end) in enumerate(self.bounds):
            if time >= start and time < end:
                gmm = self.gmms[i]
                return score_point(gmm, long, lat)

    def predict(self, time, long, lat):
        s = self.score(time, long, lat)
        if s > self.theta:
            return True
        return False

        


# def aggregate_weekends(df):
#     pass 

# def aggregate_weekdays(df):
#     pass 

# def aggregate_pattern(df):
#     weekends = df[df['weekend'] == 1]
#     weekdays = df[df['weekend'] == 0]
#     df_weekend = aggregate_weekends(weekends)
#     df_weekday = aggregate_weekdays(weekdays)
#     return df_weekday, df_weekend

# def check_for_anomaly(pattern_frame, time_threshold, distance_threshold, time, long, lat):
#     pass 

# def check_against_pattern(df_weekend, df_weekday, time, long, lat, is_weekend):
#     if is_weekend:
#         return check_for_anomaly(df_weekend, tt, dt, time, long, lat)
#     return check_for_anomaly(df_weekday, tt, dt, time, long, lat)

    