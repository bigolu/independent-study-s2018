"""
An SVC for road types.
"""
import csv
import collections
import statistics
from enum import IntEnum
import math

from sklearn import svm
import numpy as np
from dateutil import parser
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Column(IntEnum):
    DATE = 0
    TRIP_CHAR = 1
    ROAD_ID = 2
    LON = 3
    LAT = 4
    VEHICLE_TYPE = 5
    LABEL = 6


CACHE_SIZE = 1000
label_names = None


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

    # print('COS = {}'.format(cos))
    # sometimes cos > 1?
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
            vehicle_type = fname.split('/')[2]
            seen_s = False
            prev = None

            reader = tqdm(csv.reader(f, delimiter=","))
            for count, datum in enumerate(reader):
                if count >= limit:
                    break

                reader.set_description_str('Processing row {}/{} in {}'
                                           .format(count, limit, fname))

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

                prev = datum

    # map labels to ints
    global label_names
    # get a set of all unique labels then convert to a list
    # so that it is indexable
    label_names = list(set([datum[Column.LABEL] for datum in data]))
    for datum in data:
        datum[Column.LABEL] = label_names.index(
            datum[Column.LABEL])

    return data


def get_features(data):
    roads = collections.defaultdict(
        lambda: {'num_truck': 0, 'num_pv': 0, 'speeds': [],
                 'label': -1})

    prev = None
    data_tqdm = tqdm(data)
    data_len = len(data)
    for idx, datum in enumerate(data_tqdm):
        data_tqdm.set_description_str('Processing datum {}/{}'
                                      .format(idx, data_len))

        date, trip_char, road_id, lon, lat, vehicle_type, label =\
            datum

        roads[road_id]['label'] = label

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

            # things happen
            if miles == 0 or hours == 0:
                continue

            speed = miles / hours
            # print('MILES = {}, HOURS = {}, SPEED = {}'
            #       .format(miles, hours, speed))
            roads[road_id]['speeds'].append(speed)

        prev = datum

    features = []
    labels = []
    roads_len = len(roads)
    roads_tqdm = tqdm(enumerate(roads.items()))
    for i, (k, v) in roads_tqdm:
        roads_tqdm.set_description_str('Creating feature vector {}/{}'
                                       .format(i, roads_len))

        num_truck, num_pv, speeds, label = (v['num_truck'], v['num_pv'],
                                            v['speeds'], v['label'])
        speeds.sort()

        # for some reason some roads have no speeds?
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
            num_truck / (num_pv + num_truck)])
        labels.append(label)

    return (features, labels)


def get_train_test_data(features, labels):
    return [
        np.ascontiguousarray(arr, dtype=np.float32)
        for arr
        in train_test_split(features, labels, test_size=.4, random_state=0)]


def get_classifier(features, labels):
    print('Fitting model.....')
    clf = svm.SVC(CACHE_SIZE).fit(features, labels)
    print('Fitting model DONE')

    return clf


def get_score(clf, features, labels):
    print('Calculating score.....')
    score = clf.score(features, labels)
    print('Calculating score DONE')

    return score


if __name__ == '__main__':
    data = get_data(100000)
    features, labels = get_features(data)
    f_train, f_test, l_train, l_test = get_train_test_data(features, labels)
    clf = get_classifier(f_train, l_train)
    score = get_score(clf, f_test, l_test)
    print('SCORE: {}%'.format(int(score * 100)))
