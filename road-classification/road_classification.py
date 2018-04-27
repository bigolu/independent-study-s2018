"""
An SVC for road types.
"""
import sys
import csv
import collections
import statistics

from sklearn import svm
import numpy as np
from dateutil import parser


CACHE_SIZE = 1000
classifications = None


# def debug() -> None:
#     """
#     for testing
#     """
#     training_features = np.ascontiguousarray([[0, 0], [1, 1]],
#                                              dtype=np.float32)
#     training_labels = np.ascontiguousarray([0, 1], dtype=np.float32)

#     # feature scale features

#     assert training_labels.flags['C_CONTIGUOUS'], \
#         ("training labels aren't C_CONTIGUOUS")
#     assert training_features.flags['C_CONTIGUOUS'], \
#         ("training features aren't C_CONTIGUOUS")

#     road_classifier.fit(training_features, training_labels)

#     print(road_classifier.predict([[2., 2.]]))

# def main():
#     """
#     Classify roads.
#     """
#     raw_data = load_data(sys.argv[1:])
#     feature_vectors  = make_feature_vectors(raw_data)
#     labels = make_labels(raw_data)
#     road_classifier = svm.SVC(CACHE_SIZE)
#     train()


def get_data(limit):
    fnames = ['../data/pv/all.csv', '../data/truck/all.csv']
    data = []

    for fname in fnames:
        with open(fname) as f:
            count = 0
            vehicle_type = fname.split('/')[2]

            for datum in csv.reader(f, delimiter=","):
                datum = [parser.parse(datum[3]), datum[6].lower(),
                         int(datum[7]), float(datum[8]), float(datum[9]),
                         vehicle_type, datum[15]]
                data.append(datum)

                count += 1
                if count >= limit:
                    break

    # map classifications to ints
    global classifications
    classifications = list(set([datum[6] for datum in data]))
    for datum in data:
        datum[6] = classifications.index(datum[6])

    # TODO: filter out remainder of trip at beginning of data
    # (data should start with an 'S')

    # TODO: fix the extra 'S' and 'E' chars

    return data

def get_features(data):
    roads = collections.defaultdict(lambda: {'num_truck': 0, 'num_pv': 0, 'speeds': [], 'classification': -1})

    prev = None
    for datum in data:
        date, trip_char, road_id, lon, lat, vehicle_type, classification = datum

        roads[road_id]['classification'] = classification


        # add to vehicle count
        vehicle_key = 'num_truck' if vehicle_type == 'truck' else 'num_pv'
        roads[road_id][vehicle_key] += 1

        # add to speeds if not last point
        if prev and (prev[1] == 's' or prev[1] == 'm') and (trip_char == 'm' or trip_char == 'e'):
            hours = date - prev[0]
            hours = speed.seconds / 60 / 60

            latlon = LatLon(Latitude(), Longitude())
            prev_latlon = LatLon(Latitude(), Longitude())
            miles = latlon.distance(prev_latlon) * .6214 # km to mi

            speed = miles / hours
            road[road_id]['speeds'].append(speed)

        prev = datum

    features = []
    classifications = []
    for k, v in roads.items():
        num_truck, num_pv, speeds, classification = v
        speeds.sort()
        features.append([speeds[len(speeds) / 4], statistics.median(speeds),
                         speeds[(len(speeds) / 4) * 3], statistics.mean(speeds),
                         1 if num_pv > 0 else 0, 1 if num_truck > 0 else 0,
                         num_pv / (num_pv + num_truck), num_truck / (num_pv + num_truck)])
        classifications.append(classification)

    return (features, classifications)

if __name__ == '__main__':
    data = get_data(10)
    features, classifications = get_features(data)

