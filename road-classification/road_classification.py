"""
An SVC for road types.
"""
import sys
import csv
import collections
import statistics
from enum import IntEnum
import math

from sklearn import svm
import numpy as np
from dateutil import parser


class Column(IntEnum):
    DATE = 0
    TRIP_CHAR = 1
    ROAD_ID = 2
    LON = 3
    LAT = 4
    VEHICLE_TYPE = 5
    CLASSIFICATION = 6


CACHE_SIZE = 1000
classification_names = None


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

# distance between two points in miles
# see: https://www.johndcook.com/blog/python_longitude_latitude/
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2)
           + math.cos(phi1)*math.cos(phi2))

    print('COS = {}'.format(cos))
    # TODO: sometimes cos > 1?
    if cos > 1:
        # ¯\_(ツ)_/¯
        cos = 1

    arc = math.acos(cos)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * 3960


def get_data(limit):
    fnames = ['../data/pv/all.csv', '../data/truck/all.csv']
    data = []

    for fname in fnames:
        with open(fname) as f:
            count = 0
            vehicle_type = fname.split('/')[2]
            seen_s = False
            prev = None

            for datum in csv.reader(f, delimiter=","):
                datum = [parser.parse(datum[3]), datum[6].lower(),
                         int(datum[7]), float(datum[8]), float(datum[9]),
                         vehicle_type, datum[15]]

                # filter out remainder of trip at beginning of data
                # (data should start with an 's')
                if datum[Column.TRIP_CHAR] == 's':
                    seen_s = True
                if not seen_s and datum[Column.TRIP_CHAR] != 's':
                    continue

                # filter out extra 'S' and 'E' chars
                is_start_or_end = (datum[Column.TRIP_CHAR] == 's'
                                   or datum[Column.TRIP_CHAR] == 'e')
                same_char_as_prev = (prev
                                     and (prev[Column.TRIP_CHAR]
                                          == datum[Column.TRIP_CHAR]))
                if is_start_or_end and same_char_as_prev:
                    continue

                data.append(datum)

                count += 1
                if count >= limit:
                    break

                prev = datum

    # map classifications to ints
    global classification_names
    # get a set of all unique classifications then convert to a list
    # so that it is indexable
    classification_names = list(set([datum[Column.CLASSIFICATION]
                                for datum in data]))
    for datum in data:
        datum[Column.CLASSIFICATION] = classification_names.index(
            datum[Column.CLASSIFICATION])

    return data


def get_features(data):
    roads = collections.defaultdict(
        lambda: {'num_truck': 0, 'num_pv': 0, 'speeds': [],
                 'classification': -1})

    prev = None
    for datum in data:
        date, trip_char, road_id, lon, lat, vehicle_type, classification =\
            datum

        roads[road_id]['classification'] = classification

        # add to vehicle count
        vehicle_key = 'num_truck' if vehicle_type == 'truck' else 'num_pv'
        roads[road_id][vehicle_key] += 1

        # add to speeds if not last point
        part_of_same_trip = (prev
                             and (prev[Column.TRIP_CHAR] == 's'
                                  or prev[Column.TRIP_CHAR] == 'm')
                             and (trip_char == 'm' or trip_char == 'e'))
        if part_of_same_trip:
            time_diff = date - prev[Column.DATE]
            hours = time_diff.seconds / 60 / 60

            prev_lat = prev[Column.LAT]
            prev_lon = prev[Column.LON]
            # print('LAT = {}, LON = {}, PREVLAT = {}, PREVLON = {}'
            #       .format(lat, lon, prev_lat, prev_lon))
            miles = distance_on_unit_sphere(lat, lon, prev_lat,
                                            prev_lon)

            speed = miles / hours
            # print('MILES = {}, HOURS = {}, SPEED = {}'
            #       .format(miles, hours, speed))
            roads[road_id]['speeds'].append(speed)

        prev = datum

    features = []
    classifications = []
    for k, v in roads.items():
        num_truck, num_pv, speeds, classification = (v['num_truck'],
                                                     v['num_pv'], v['speeds'],
                                                     v['classification'])
        speeds.sort()

        # TODO: for some reason some roads have no speeds?
        if not speeds:
            continue

        features.append([
            # 1/4 speed
            speeds[int(len(speeds) / 4)],
            # median speed
            statistics.median(speeds),
            # 3/4 speed
            speeds[int((len(speeds) / 4) * 3)],
            # average speed
            statistics.mean(speeds),
            # existance of pv
            1 if num_pv > 0 else 0,
            # existance of truck
            1 if num_truck > 0 else 0,
            # percentage of pv on road
            num_pv / (num_pv + num_truck),
            # percentage of truck on road
            num_truck / (num_pv + num_truck)
        ])
        classifications.append(classification)

    return (features, classifications)


if __name__ == '__main__':
    data = get_data(1000)
    features, classifications = get_features(data)
    print(features)
    print('-------------------------------------------------')
    print(classifications)
    print('-------------------------------------------------')
    print(classification_names)
