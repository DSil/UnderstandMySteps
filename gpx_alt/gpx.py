# -*- coding: utf-8 -*-

# Copyright 2011 Tomo Krajina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPX related stuff
"""

import pdb

import logging as mod_logging
import math as mod_math
import collections as mod_collections
import copy as mod_copy
import datetime as mod_datetime
import numpy
import datetime

from . import utils as mod_utils
from . import geo as mod_geo
from . import gpxfield as mod_gpxfield
from datetime import timedelta
# GPX date format to be used when writing the GPX output:
DATE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

# GPX date format(s) used for parsing. The T between date and time and Z after
# time are allowed, too:
DATE_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    #'%Y-%m-%d %H:%M:%S%z',
    #'%Y-%m-%d %H:%M:%S.%f%z',
]


# When computing stopped time -- this is the minimum speed between two points,
# if speed is less than this value -- we'll assume it is zero
DEFAULT_STOPPED_SPEED_THRESHOLD = 1

GPX_10_POINT_FIELDS = [
        mod_gpxfield.GPXField('latitude', attribute='lat', type=mod_gpxfield.FLOAT_TYPE, mandatory=True),
        mod_gpxfield.GPXField('longitude', attribute='lon', type=mod_gpxfield.FLOAT_TYPE, mandatory=True),
        mod_gpxfield.GPXField('elevation', 'ele', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('time', type=mod_gpxfield.TIME_TYPE),
        mod_gpxfield.GPXField('magnetic_variation', 'magvar', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('geoid_height', 'geoidheight', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('name'),
        mod_gpxfield.GPXField('comment', 'cmt'),
        mod_gpxfield.GPXField('description', 'desc'),
        mod_gpxfield.GPXField('source', 'src'),
        mod_gpxfield.GPXField('link', 'url'),
        mod_gpxfield.GPXField('link_text', 'urlname'),
        mod_gpxfield.GPXField('symbol', 'sym'),
        mod_gpxfield.GPXField('type'),
        mod_gpxfield.GPXField('type_of_gpx_fix', 'fix', possible=('none', '2d', '3d', 'dgps', 'pps',)),
        mod_gpxfield.GPXField('satellites', 'sat', type=mod_gpxfield.INT_TYPE),
        mod_gpxfield.GPXField('horizontal_dilution', 'hdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('vertical_dilution', 'vdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('position_dilution', 'pdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('age_of_dgps_data', 'ageofdgpsdata', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('dgps_id', 'dgpsid'),
]
GPX_11_POINT_FIELDS = [
        mod_gpxfield.GPXField('latitude', attribute='lat', type=mod_gpxfield.FLOAT_TYPE, mandatory=True),
        mod_gpxfield.GPXField('longitude', attribute='lon', type=mod_gpxfield.FLOAT_TYPE, mandatory=True),
        mod_gpxfield.GPXField('elevation', 'ele', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('time', type=mod_gpxfield.TIME_TYPE),
        mod_gpxfield.GPXField('magnetic_variation', 'magvar', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('geoid_height', 'geoidheight', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('name'),
        mod_gpxfield.GPXField('comment', 'cmt'),
        mod_gpxfield.GPXField('description', 'desc'),
        mod_gpxfield.GPXField('source', 'src'),
        'link',
            mod_gpxfield.GPXField('link', attribute='href'),
            mod_gpxfield.GPXField('link_text', tag='text'),
            mod_gpxfield.GPXField('link_type', tag='type'),
        '/link',
        mod_gpxfield.GPXField('symbol', 'sym'),
        mod_gpxfield.GPXField('type'),
        mod_gpxfield.GPXField('type_of_gpx_fix', 'fix', possible=('none', '2d', '3d', 'dgps', 'pps',)),
        mod_gpxfield.GPXField('satellites', 'sat', type=mod_gpxfield.INT_TYPE),
        mod_gpxfield.GPXField('horizontal_dilution', 'hdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('vertical_dilution', 'vdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('position_dilution', 'pdop', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('age_of_dgps_data', 'ageofdgpsdata', type=mod_gpxfield.FLOAT_TYPE),
        mod_gpxfield.GPXField('dgps_id', 'dgpsid'),
        mod_gpxfield.GPXExtensionsField('extensions'),
]

# GPX1.0 track points have two more fields after time
# Note that this is not true for GPX1.1
GPX_TRACK_POINT_FIELDS = GPX_10_POINT_FIELDS[:4] \
        + [ \
                mod_gpxfield.GPXField('course', type=mod_gpxfield.FLOAT_TYPE), \
                mod_gpxfield.GPXField('speed', type=mod_gpxfield.FLOAT_TYPE) \
          ] \
        + GPX_10_POINT_FIELDS[4:]

# When possible, the result of various methods are named tuples defined here:
Bounds = mod_collections.namedtuple(
    'Bounds',
    ('min_latitude', 'max_latitude', 'min_longitude', 'max_longitude'))
TimeBounds = mod_collections.namedtuple(
    'TimeBounds',
    ('start_time', 'end_time'))
MovingData = mod_collections.namedtuple(
    'MovingData',
    ('moving_time', 'stopped_time', 'moving_distance', 'stopped_distance', 'max_speed'))
UphillDownhill = mod_collections.namedtuple(
    'UphillDownhill',
    ('uphill', 'downhill'))
MinimumMaximum = mod_collections.namedtuple(
    'MinimumMaximum',
    ('minimum', 'maximum'))
NearestLocationData = mod_collections.namedtuple(
    'NearestLocationData',
    ('location', 'track_no', 'segment_no', 'point_no'))
PointData = mod_collections.namedtuple(
    'PointData',
    ('point', 'distance_from_start', 'track_no', 'segment_no', 'point_no'))


class GPXException(Exception):
    """
    Exception used for invalid GPX files. Is is used when the XML file is
    valid but something is wrong with the GPX data.
    """
    pass


class GPXXMLSyntaxException(GPXException):
    """
    Exception used when the the XML syntax is invalid.

    The __cause__ can be a minidom or lxml exception (See http://www.python.org/dev/peps/pep-3134/).
    """
    def __init__(self, message, original_exception):
        GPXException.__init__(self, message)
        self.__cause__ = original_exception


class GPXWaypoint(mod_geo.Location):
    time = None
    name = None
    description = None
    symbol = None
    type = None
    comment = None

    # Horizontal dilution of precision
    horizontal_dilution = None
    # Vertical dilution of precision
    vertical_dilution = None
    # Position dilution of precision
    position_dilution = None

    def __init__(self, latitude, longitude, elevation=None, time=None,
                 name=None, description=None, symbol=None, type=None,
                 comment=None, horizontal_dilution=None, vertical_dilution=None,
                 position_dilution=None):
        mod_geo.Location.__init__(self, latitude, longitude, elevation)

        self.time = time
        self.name = name
        self.description = description
        self.symbol = symbol
        self.type = type
        self.comment = comment

        self.horizontal_dilution = horizontal_dilution
        self.vertical_dilution = vertical_dilution
        self.position_dilution = position_dilution

    def __str__(self):
        return '[wpt{%s}:%s,%s@%s]' % (self.name, self.latitude, self.longitude, self.elevation)

    def __repr__(self):
        representation = '%s, %s' % (self.latitude, self.longitude)
        for attribute in 'elevation', 'time', 'name', 'description', 'symbol', 'type', 'comment', \
                'horizontal_dilution', 'vertical_dilution', 'position_dilution':
            value = getattr(self, attribute)
            if value is not None:
                representation += ', %s=%s' % (attribute, repr(value))
        return 'GPXWaypoint(%s)' % representation

    def to_xml(self, version=None):
        content = ''
        if self.elevation is not None:
            content += mod_utils.to_xml('ele', content=self.elevation)
        if self.time:
            content += mod_utils.to_xml('time', content=self.time.strftime(DATE_FORMAT))
        if self.name:
            content += mod_utils.to_xml('name', content=self.name, escape=True)
        if self.description:
            content += mod_utils.to_xml('desc', content=self.description, escape=True)
        if self.symbol:
            content += mod_utils.to_xml('sym', content=self.symbol, escape=True)
        if self.type:
            content += mod_utils.to_xml('type', content=self.type, escape=True)

        if version == '1.1':  # TODO
            content += mod_utils.to_xml('cmt', content=self.comment, escape=True)

        if self.horizontal_dilution:
            content += mod_utils.to_xml('hdop', content=self.horizontal_dilution)
        if self.vertical_dilution:
            content += mod_utils.to_xml('vdop', content=self.vertical_dilution)
        if self.position_dilution:
            content += mod_utils.to_xml('pdop', content=self.position_dilution)

        return mod_utils.to_xml('wpt', attributes={'lat': self.latitude, 'lon': self.longitude}, content=content)

    def get_max_dilution_of_precision(self):
        """
        Only care about the max dop for filtering, no need to go into too much detail
        """
        return max(self.horizontal_dilution, self.vertical_dilution, self.position_dilution)

    def __hash__(self):
        return mod_utils.hash_object(self, 'time', 'name', 'description', 'symbol', 'type',
                                     'comment', 'horizontal_dilution', 'vertical_dilution', 'position_dilution')


class GPXRoute:
    def __init__(self, name=None, description=None, number=None):
        self.name = name
        self.description = description
        self.number = number

        self.points = []

    def remove_elevation(self):
        for point in self.points:
            point.remove_elevation()

    def length(self):
        return mod_geo.length_2d(self.points)

    def get_center(self):
        if not self.points:
            return None

        if not self.points:
            return None

        sum_lat = 0.
        sum_lon = 0.
        n = 0.

        for point in self.points:
            n += 1.
            sum_lat += point.latitude
            sum_lon += point.longitude

        if not n:
            return mod_geo.Location(float(0), float(0))

        return mod_geo.Location(latitude=sum_lat / n, longitude=sum_lon / n)

    def walk(self, only_points=False):
        for point_no, point in enumerate(self.points):
            if only_points:
                yield point
            else:
                yield point, point_no

    def get_points_no(self):
        return len(self.points)

    def move(self, location_delta):
        for route_point in self.points:
            route_point.move(location_delta)

    def to_xml(self, version=None):
        content = ''
        if self.name:
            content += mod_utils.to_xml('name', content=self.name, escape=True)
        if self.description:
            content += mod_utils.to_xml('desc', content=self.description, escape=True)
        if self.number:
            content += mod_utils.to_xml('number', content=self.number)
        for route_point in self.points:
            content += route_point.to_xml(version)

        return mod_utils.to_xml('rte', content=content)

    def __hash__(self):
        return mod_utils.hash_object(self, 'name', 'description', 'number', 'points')

    def __repr__(self):
        representation = ''
        for attribute in 'name', 'description', 'number':
            value = getattr(self, attribute)
            if value is not None:
                representation += '%s%s=%s' % (', ' if representation else '', attribute, repr(value))
        representation += '%spoints=[%s])' % (', ' if representation else '', '...' if self.points else '')
        return 'GPXRoute(%s)' % representation


class GPXRoutePoint(mod_geo.Location):
    def __init__(self, latitude, longitude, elevation=None, time=None, name=None,
                 description=None, symbol=None, type=None, comment=None,
                 horizontal_dilution=None, vertical_dilution=None,
                 position_dilution=None):

        mod_geo.Location.__init__(self, latitude, longitude, elevation)

        self.time = time
        self.name = name
        self.description = description
        self.symbol = symbol
        self.type = type
        self.comment = comment

        self.horizontal_dilution = horizontal_dilution  # Horizontal dilution of precision
        self.vertical_dilution = vertical_dilution      # Vertical dilution of precision
        self.position_dilution = position_dilution      # Position dilution of precision

    def __str__(self):
        return '[rtept{%s}:%s,%s@%s]' % (self.name, self.latitude, self.longitude, self.elevation)

    def __repr__(self):
        representation = '%s, %s' % (self.latitude, self.longitude)
        for attribute in 'elevation', 'time', 'name', 'description', 'symbol', 'type', 'comment', \
                'horizontal_dilution', 'vertical_dilution', 'position_dilution':
            value = getattr(self, attribute)
            if value is not None:
                representation += ', %s=%s' % (attribute, repr(value))
        return 'GPXRoutePoint(%s)' % representation

    def to_xml(self, version=None):
        content = ''
        if self.elevation is not None:
            content += mod_utils.to_xml('ele', content=self.elevation)
        if self.time:
            content += mod_utils.to_xml('time', content=self.time.strftime(DATE_FORMAT))
        if self.name:
            content += mod_utils.to_xml('name', content=self.name, escape=True)
        if self.comment:
            content += mod_utils.to_xml('cmt', content=self.comment, escape=True)
        if self.description:
            content += mod_utils.to_xml('desc', content=self.description, escape=True)
        if self.symbol:
            content += mod_utils.to_xml('sym', content=self.symbol, escape=True)
        if self.type:
            content += mod_utils.to_xml('type', content=self.type, escape=True)

        if self.horizontal_dilution:
            content += mod_utils.to_xml('hdop', content=self.horizontal_dilution)
        if self.vertical_dilution:
            content += mod_utils.to_xml('vdop', content=self.vertical_dilution)
        if self.position_dilution:
            content += mod_utils.to_xml('pdop', content=self.position_dilution)

        return mod_utils.to_xml('rtept', attributes={'lat': self.latitude, 'lon': self.longitude}, content=content)

    def __hash__(self):
        return mod_utils.hash_object(self, 'time', 'name', 'description', 'symbol', 'type', 'comment',
                                     'horizontal_dilution', 'vertical_dilution', 'position_dilution')


class GPXTrackPoint(mod_geo.Location):
    gpx_10_fields = GPX_TRACK_POINT_FIELDS
    gpx_11_fields = GPX_11_POINT_FIELDS

    __slots__ = ('latitude', 'longitude', 'elevation', 'time', 'course', 
                 'speed', 'magnetic_variation', 'geoid_height', 'name', 
                 'comment', 'description', 'source', 'link', 'link_text', 
                 'symbol', 'type', 'type_of_gpx_fix', 'satellites', 
                 'horizontal_dilution', 'vertical_dilution', 
                 'position_dilution', 'age_of_dgps_data', 'dgps_id', 
                 'link_type', 'extensions')
    
    def __init__(self, latitude, longitude, elevation=None, time=None, symbol=None, comment=None,
                 horizontal_dilution=None, vertical_dilution=None, position_dilution=None, speed=None,
                 name=None):
        mod_geo.Location.__init__(self, latitude, longitude, elevation)

        self.time = time
        self.symbol = symbol
        self.comment = comment
        self.name = name

        self.horizontal_dilution = horizontal_dilution  # Horizontal dilution of precision
        self.vertical_dilution = vertical_dilution      # Vertical dilution of precision
        self.position_dilution = position_dilution      # Position dilution of precision

        self.speed = speed
        
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.course = None
        self.magnetic_variation = None
        self.geoid_height = None
        self.description = None
        self.source = None
        self.link = None
        self.link_text = None
        self.link_type = None
        self.type = None
        self.type_of_gpx_fix = None
        self.satellites = None
        self.age_of_dgps_data = None
        self.dgps_id = None
        self.extensions = None        

    def __repr__(self):
        representation = '%s, %s' % (self.latitude, self.longitude)
        for attribute in 'elevation', 'time', 'symbol', 'comment', 'horizontal_dilution', \
                'vertical_dilution', 'position_dilution', 'speed', 'name':
            value = getattr(self, attribute)
            if value is not None:
                representation += ', %s=%s' % (attribute, repr(value))
        return 'GPXTrackPoint(%s)' % representation
    
    def __eq__(self, p):
        return isinstance(p, GPXTrackPoint) and\
               p.latitude == self.latitude and p.longitude == self.longitude        

    def adjust_time(self, delta):
        """
        Add the amount of time in delta to the point's time. Adjust with a negative delta in order to subtract
        time from the point.
        """
        if self.time:
            self.time += delta

    def remove_time(self):
        """ Will remove time metadata. """
        self.time = None

    def to_xml(self, version=None):
        content = ''

        if self.elevation is not None:
            content += mod_utils.to_xml('ele', content=self.elevation)
        if self.time:
            content += mod_utils.to_xml('time', content=self.time.strftime(DATE_FORMAT))
        if self.comment:
            content += mod_utils.to_xml('cmt', content=self.comment, escape=True)
        if self.name:
            content += mod_utils.to_xml('name', content=self.name, escape=True)
        if self.symbol:
            content += mod_utils.to_xml('sym', content=self.symbol, escape=True)

        if self.horizontal_dilution:
            content += mod_utils.to_xml('hdop', content=self.horizontal_dilution)
        if self.vertical_dilution:
            content += mod_utils.to_xml('vdop', content=self.vertical_dilution)
        if self.position_dilution:
            content += mod_utils.to_xml('pdop', content=self.position_dilution)

        if self.speed:
            content += mod_utils.to_xml('speed', content=self.speed)

        return mod_utils.to_xml('trkpt', {'lat': self.latitude, 'lon': self.longitude}, content=content)

    def time_difference(self, track_point):
        """ Time distance in seconds between times of those two points """
        if not self.time or not track_point or not track_point.time:
            return None

        time_1 = self.time
        time_2 = track_point.time

        if time_1 == time_2:
            return 0

        if time_1 > time_2:
            delta = time_1 - time_2
        else:
            delta = time_2 - time_1

        return mod_utils.total_seconds(delta)

    def speed_between(self, track_point):
        """
        Note that this is a *computed* speed. The self.speed is the value
        specified in the GPX file.
        """
        if not track_point:
            return None

        seconds = self.time_difference(track_point)
        length = self.distance_3d(track_point)
        if not length:
            length = self.distance_2d(track_point)

        if not seconds or not length:
            return None

        return length / float(seconds)

    def __str__(self):
        return '[trkpt:%s,%s@%s@%s]' % (self.latitude, self.longitude, self.elevation, self.time)

    def __hash__(self):
        return mod_utils.hash_object(self, 'latitude', 'longitude', 'elevation', 'time', 'symbol', 'comment',
                                     'horizontal_dilution', 'vertical_dilution', 'position_dilution', 'speed')

class GPXTrackSegment:
    gpx_10_fields = [
            mod_gpxfield.GPXComplexField('points', tag='trkpt', classs=GPXTrackPoint, is_list=True),
            mod_gpxfield.GPXField('description', 'desc'),
    ]
    gpx_11_fields = [
            mod_gpxfield.GPXComplexField('points', tag='trkpt', classs=GPXTrackPoint, is_list=True),
            mod_gpxfield.GPXExtensionsField('extensions'),
            mod_gpxfield.GPXField('description', 'desc'),
    ]

    __slots__ = ('points', 'extensions', 'description', )
    
    def getKey(self):
        return self.points[0].time

    def __cmp__(self, other):
        if hasattr(other, 'getKey'):
            return cmp(self.getKey() , other.getKey())

    def __init__(self, points=None):
        self.points = points if points else []
        self.extensions = None
        self.description = None

    def simplify(self, max_distance=None, max_time=None):
        """
        Simplify using the Ramer-Douglas-Peucker algorithm: http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        """
        if not max_distance:
            max_distance = 10

        self.points = mod_geo.simplify_polyline(self.points, max_distance, max_time)

    def simplify2(self, max_points):
        """
        Simplify using the Ramer-Douglas-Peucker algorithm: http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        """
        self.points = mod_geo.simplify2(self.points, max_points)
        
    def simplify3(self):
        
        avg_speed = self.length_3d() / self.get_duration()
        
        self.points = mod_geo.simplify3(self.points, avg_speed)

    def reduce_points(self, min_distance, min_time):
        reduced_points = []
        for point in self.points:
            if reduced_points:
                distance = reduced_points[-1].distance_3d(point)
                time = point.time-reduced_points[-1].time
                #print time
                if distance >= min_distance or time >= timedelta(seconds=min_time):
                    reduced_points.append(point)
            else:
                # Leave first point:
                reduced_points.append(point)

        self.points = reduced_points

    def _find_next_simplified_point(self, pos, max_distance):
        for candidate in range(pos + 1, len(self.points) - 1):
            for i in range(pos + 1, candidate):
                d = mod_geo.distance_from_line(self.points[i],
                                               self.points[pos],
                                               self.points[candidate])
                if d > max_distance:
                    return candidate - 1
        return None

    def adjust_time(self, delta):
        for track_point in self.points:
            track_point.adjust_time(delta)

    def remove_time(self):
        for track_point in self.points:
            track_point.remove_time()

    def remove_elevation(self):
        for track_point in self.points:
            track_point.remove_elevation()

    def length_2d(self):
        return mod_geo.length_2d(self.points)

    def length_3d(self):
        return mod_geo.length_3d(self.points)

    def move(self, location_delta):
        for track_point in self.points:
            track_point.move(location_delta)

    def walk(self, only_points=False):
        """ Use this to iterate through points """
        for point_no, point in enumerate(self.points):
            if only_points:
                yield point
            else:
                yield point, point_no

    def get_points_no(self):
        """ Number of points """
        if not self.points:
            return 0
        return len(self.points)

    def split(self, first_point, last_point):
        """ Splits this segment in two parts. Point #point_no remains in the first part.
        Returns a list with two GPXTrackSegments """
        part_1 = self.points[first_point:last_point]
        #part_2 = self.points[point_no + 1:]
        return GPXTrackSegment(part_1)#, GPXTrackSegment(part_2)

    def join(self, track_segment):
        """ Joins with another segment """
        self.points += track_segment.points

    def remove_point(self, point_no):
        if point_no < 0 or point_no >= len(self.points):
            return

        part_1 = self.points[:point_no]
        part_2 = self.points[point_no + 1:]

        self.points = part_1 + part_2

    def get_moving_data(self, stopped_speed_threshold=None):
        if not stopped_speed_threshold:
            stopped_speed_threshold = DEFAULT_STOPPED_SPEED_THRESHOLD

        moving_time = 0.
        stopped_time = 0.

        moving_distance = 0.
        stopped_distance = 0.

        speeds_and_distances = []

        for i in range(1, len(self.points)):

            previous = self.points[i - 1]
            point = self.points[i]

            # Won't compute max_speed for first and last because of common GPS
            # recording errors, and because smoothing don't work well for those
            # points:
            first_or_last = i in [0, 1, len(self.points) - 1]
            if point.time and previous.time:
                timedelta = point.time - previous.time

                if point.elevation and previous.elevation:
                    distance = point.distance_3d(previous)
                else:
                    distance = point.distance_2d(previous)

                seconds = mod_utils.total_seconds(timedelta)
                speed_kmh = 0
                if seconds > 0:
                    # TODO: compute treshold in m/s instead this to kmh every time:
                    speed_kmh = (distance / 1000.) / (mod_utils.total_seconds(timedelta) / 60. ** 2)

                #print speed, stopped_speed_threshold
                if speed_kmh <= stopped_speed_threshold:
                    stopped_time += mod_utils.total_seconds(timedelta)
                    stopped_distance += distance
                else:
                    moving_time += mod_utils.total_seconds(timedelta)
                    moving_distance += distance

                    if distance and moving_time:
                        speeds_and_distances.append((distance / mod_utils.total_seconds(timedelta), distance, ))

        max_speed = None
        if speeds_and_distances:
            max_speed = mod_geo.calculate_max_speed(speeds_and_distances)

        return MovingData(moving_time, stopped_time, moving_distance, stopped_distance, max_speed)

    def get_time_bounds(self):
        start_time = None
        end_time = None

        for point in self.points:
            if point.time:
                if not start_time:
                    start_time = point.time
                if point.time:
                    end_time = point.time

        return TimeBounds(start_time, end_time)

    def get_bounds(self):
        min_lat = None
        max_lat = None
        min_lon = None
        max_lon = None

        for point in self.points:
            if min_lat is None or point.latitude < min_lat:
                min_lat = point.latitude
            if max_lat is None or point.latitude > max_lat:
                max_lat = point.latitude
            if min_lon is None or point.longitude < min_lon:
                min_lon = point.longitude
            if max_lon is None or point.longitude > max_lon:
                max_lon = point.longitude

        return Bounds(min_lat, max_lat, min_lon, max_lon)

    def get_speed(self, point_no):
        """ Get speed at that point. Point may be a GPXTrackPoint instance or integer (point index) """

        point = self.points[point_no]

        previous_point = None
        next_point = None

        if 0 < point_no < len(self.points):
            previous_point = self.points[point_no - 1]
        if 0 < point_no < len(self.points) - 1:
            next_point = self.points[point_no + 1]

        #mod_logging.debug('previous: %s' % previous_point)
        #mod_logging.debug('next: %s' % next_point)

        speed_1 = point.speed_between(previous_point)
        speed_2 = point.speed_between(next_point)

        if speed_1:
            speed_1 = abs(speed_1)
        if speed_2:
            speed_2 = abs(speed_2)

        if speed_1 and speed_2:
            return (speed_1 + speed_2) / 2.

        if speed_1:
            return speed_1

        return speed_2

    def add_elevation(self, delta):
        mod_logging.debug('delta = %s' % delta)

        if not delta:
            return

        for track_point in self.points:
            if track_point.elevation is not None:
                track_point.elevation += delta

    def add_missing_data(self, get_data_function, add_missing_function):
        if not get_data_function:
            raise GPXException('Invalid get_data_function: %s' % get_data_function)
        if not add_missing_function:
            raise GPXException('Invalid add_missing_function: %s' % add_missing_function)

        # Points between two points *without* data:
        interval = []
        # Points before and after the interval *with* data:
        start_point = None

        previous_point = None
        for track_point in self.points:
            data = get_data_function(track_point)
            if data is None and previous_point:
                if not start_point:
                    start_point = previous_point
                interval.append(track_point)
            else:
                if interval:
                    distances_ratios = self._get_interval_distances_ratios(interval,
                                                                           start_point, track_point)
                    add_missing_function(interval, start_point, track_point,
                                         distances_ratios)
                    start_point = None
                    interval = []
            previous_point = track_point

    def _get_interval_distances_ratios(self, interval, start, end):
        assert start, start
        assert end, end
        assert interval, interval
        assert len(interval) > 0, interval

        distances = []
        distance_from_start = 0
        previous_point = start
        for point in interval:
            distance_from_start += float(point.distance_3d(previous_point))
            distances.append(distance_from_start)
            previous_point = point

        from_start_to_end = distances[-1] + interval[-1].distance_3d(end)

        assert len(interval) == len(distances)

        return list(map(
                lambda distance: (distance / from_start_to_end) if from_start_to_end else 0,
                distances))

    def get_duration(self):
        """ Duration in seconds """
        if not self.points or len(self.points) < 2:
            return 0

        # Search for start:
        first = self.points[0]
        if not first.time:
            first = self.points[1]

        last = self.points[-1]
        if not last.time:
            last = self.points[-2]

        if not last.time or not first.time:
            mod_logging.debug('Can\'t find time')
            return None

        if last.time < first.time:
            mod_logging.debug('Not enough time data')
            return None

        return mod_utils.total_seconds(last.time - first.time)

    def get_uphill_downhill(self):
        """
        Returns (uphill, downhill). If elevation for some points is not found
        those are simply ignored.
        """
        if not self.points:
            return UphillDownhill(0, 0)

        elevations = list(map(lambda point: point.elevation, self.points))
        uphill, downhill = mod_geo.calculate_uphill_downhill(elevations)

        return UphillDownhill(uphill, downhill)

    def get_elevation_extremes(self):
        """ return (min_elevation, max_elevation) """

        if not self.points:
            return MinimumMaximum(None, None)

        elevations = map(lambda location: location.elevation, self.points)
        elevations = filter(lambda elevation: elevation is not None, elevations)
        elevations = list(elevations)

        if len(elevations) == 0:
            return MinimumMaximum(None, None)

        return MinimumMaximum(max(elevations), min(elevations))

    def get_location_at(self, time):
        """
        Gets approx. location at given time. Note that, at the moment this method returns
        an instance of GPXTrackPoint in the future -- this may be a mod_geo.Location instance
        with approximated latitude, longitude and elevation!
        """
        if not self.points:
            return None

        if not time:
            return None

        first_time = self.points[0].time
        last_time = self.points[-1].time

        if not first_time and not last_time:
            mod_logging.debug('No times for track segment')
            return None

        if not first_time <= time <= last_time:
            mod_logging.debug('Not in track (search for:%s, start:%s, end:%s)' % (time, first_time, last_time))
            return None

        for point in self.points:
            if point.time and time <= point.time:
                # TODO: If between two points -- approx position!
                # return mod_geo.Location(point.latitude, point.longitude)
                return point

    def to_xml(self, version=None):
        content = ''
        for track_point in self.points:
            content += track_point.to_xml(version)
        return mod_utils.to_xml('trkseg', content=content)

    def get_nearest_location(self, location):
        """ Return the (location, track_point_no) on this track segment """
        if not self.points:
            return None, None

        result = None
        current_distance = None
        result_track_point_no = None
        for i in range(len(self.points)):
            track_point = self.points[i]
            if not result:
                result = track_point
            else:
                distance = track_point.distance_2d(location)
                #print current_distance, distance
                if not current_distance or distance < current_distance:
                    current_distance = distance
                    result = track_point
                    result_track_point_no = i

        return result, result_track_point_no

    def smooth(self, remove_extremes, how_much_to_smooth):
        if len(self.points) <= 3:
            return

        latitudes = []
        longitudes = []

        for point in self.points:
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)

        avg_distance = 0
        if remove_extremes:
            # compute the average distance between two points:
            distances = []
            for i in range(len(self.points))[1:]:
                distances.append(self.points[i].distance_2d(self.points[i - 1]))
            if distances:
                avg_distance = 1.0 * sum(distances) / len(distances)

        # If The point moved more than this number * the average distance between two
        # points -- then is a candidate for deletion:
        remove_2d_extremes_threshold = how_much_to_smooth * avg_distance

        new_track_points = [self.points[0]]

        for i in range(len(self.points))[2:-2]:
            new_point = None
            point_removed = False

            old_latitude = self.points[i].latitude
            new_latitude = numpy.median(latitudes[i - 2:i + 2])
            old_longitude = self.points[i].longitude
            new_longitude = numpy.median(longitudes[i - 2:i + 2])

            # TODO: This is not ideal.. Because if there are points A, B and C on the same
            # line but B is very close to C... This would remove B (and possibly) A even though
            # it is not an extreme. This is the reason for this algorithm:
            d1 = mod_geo.distance(latitudes[i - 1], longitudes[i - 1], None, latitudes[i], longitudes[i], None)
            d2 = mod_geo.distance(latitudes[i + 1], longitudes[i + 1], None, latitudes[i], longitudes[i], None)
            d = mod_geo.distance(latitudes[i - 1], longitudes[i - 1], None, latitudes[i + 1], longitudes[i + 1], None)

            #print d1, d2, d, remove_extremes

            if d1 + d2 > d * 1.05 and remove_extremes:
                d = mod_geo.distance(old_latitude, old_longitude, None, new_latitude, new_longitude, None)
                #print "d, threshold = ", d, remove_2d_extremes_threshold
                if d < remove_2d_extremes_threshold:
                    new_point = self.points[i]
                else:
                    #print 'removed 2d'
                    point_removed = True
            else:
                new_point = self.points[i]

            if new_point and not point_removed:
                new_track_points.append(new_point)

            if remove_extremes:
                self.points[i].latitude = new_latitude
                self.points[i].longitude = new_longitude

        new_track_points.append(self.points[- 1])

        #print 'len=', len(new_track_points)

        self.points = new_track_points

    def has_times(self):
        """
        Returns if points in this segment contains timestamps.

        The first point, the last point, and 75% of the points must have times for this
        method to return true.
        """
        if not self.points:
            return True
            # ... or otherwise one empty track segment would change the entire
            # track's "has_times" status!

        found = 0
        for track_point in self.points:
            if track_point.time:
                found += 1

        return len(self.points) > 2 and float(found) / float(len(self.points)) > .75

    def has_elevations(self):
        """
        Returns if points in this segment contains timestamps.

        The first point, the last point, and at least 75% of the points must have times for this
        method to return true.
        """
        if not self.points:
            return True
            # ... or otherwise one empty track segment would change the entire
            # track's "has_times" status!

        found = 0
        for track_point in self.points:
            if track_point.elevation:
                found += 1

        return len(self.points) > 2 and float(found) / float(len(self.points)) > .75

    def __hash__(self):
        return mod_utils.hash_object(self, 'points')

    def __repr__(self):
        return 'GPXTrackSegment(points=[%s])' % ('...' if self.points else '')

    def clone(self):
        return mod_copy.deepcopy(self)

    def __len__(self):
        return len(self.points)


class GPXTrack:
    gpx_10_fields = [
            mod_gpxfield.GPXField('name'),
            mod_gpxfield.GPXField('comment', 'cmt'),
            mod_gpxfield.GPXField('description', 'desc'),
            mod_gpxfield.GPXField('source', 'src'),
            mod_gpxfield.GPXField('link', 'url'),
            mod_gpxfield.GPXField('link_text', 'urlname'),
            mod_gpxfield.GPXField('number', type=mod_gpxfield.INT_TYPE),
            mod_gpxfield.GPXComplexField('segments', tag='trkseg', classs=GPXTrackSegment, is_list=True),
    ]
    gpx_11_fields = [
            mod_gpxfield.GPXField('name'),
            mod_gpxfield.GPXField('comment', 'cmt'),
            mod_gpxfield.GPXField('description', 'desc'),
            mod_gpxfield.GPXField('source', 'src'),
            'link',
                mod_gpxfield.GPXField('link', attribute='href'),
                mod_gpxfield.GPXField('link_text', tag='text'),
                mod_gpxfield.GPXField('link_type', tag='type'),
            '/link',
            mod_gpxfield.GPXField('number', type=mod_gpxfield.INT_TYPE),
            mod_gpxfield.GPXField('type'),
            mod_gpxfield.GPXExtensionsField('extensions'),
            mod_gpxfield.GPXComplexField('segments', tag='trkseg', classs=GPXTrackSegment, is_list=True),
    ]

    __slots__ = ('name', 'comment', 'description', 'source', 'link', 
                 'link_text', 'number', 'segments', 'link_type', 'type', 
                 'extensions')
    
    def __init__(self, name=None, description=None, number=None):
        self.name = name
        self.description = description
        self.number = number
        # Type is not exactly part of the standard but a is "proposed" and a 
        # lot of application use it:
        self.type = None
        
        self.comment = None
        self.source = None
        self.link = None
        self.link_text = None
        self.link_type = None
        self.extensions = None        

        self.segments = []

    def __len__(self):
        return len(self.segments)

    def simplify(self, max_distance=None):
        """
        Simplify using the Ramer-Douglas-Peucker algorithm: http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        """
        for segment in self.segments:
            segment.simplify(max_distance=max_distance)

    def reduce_points(self, min_distance):
        for segment in self.segments:
            segment.reduce_points(min_distance)

    def adjust_time(self, delta):
        for segment in self.segments:
            segment.adjust_time(delta)

    def remove_time(self):
        for segment in self.segments:
            segment.remove_time()

    def remove_elevation(self):
        for segment in self.segments:
            segment.remove_elevation()

    def remove_empty(self):
        """ Removes empty segments and/or routes """
        result = []

        for segment in self.segments:
            if len(segment.points) > 0:
                result.append(segment)

        self.segments = result

    def length_2d(self):
        length = 0
        for track_segment in self.segments:
            d = track_segment.length_2d()
            if d:
                length += d
        return length

    def get_time_bounds(self):
        start_time = None
        end_time = None

        for track_segment in self.segments:
            point_start_time, point_end_time = track_segment.get_time_bounds()
            if not start_time and point_start_time:
                start_time = point_start_time
            if point_end_time:
                end_time = point_end_time

        return TimeBounds(start_time, end_time)

    def get_bounds(self):
        min_lat = None
        max_lat = None
        min_lon = None
        max_lon = None
        for track_segment in self.segments:
            bounds = track_segment.get_bounds()

            if not mod_utils.is_numeric(min_lat) or (bounds.min_latitude and bounds.min_latitude < min_lat):
                min_lat = bounds.min_latitude
            if not mod_utils.is_numeric(max_lat) or (bounds.max_latitude and bounds.max_latitude > max_lat):
                max_lat = bounds.max_latitude
            if not mod_utils.is_numeric(min_lon) or (bounds.min_longitude and bounds.min_longitude < min_lon):
                min_lon = bounds.min_longitude
            if not mod_utils.is_numeric(max_lon) or (bounds.max_longitude and bounds.max_longitude > max_lon):
                max_lon = bounds.max_longitude

        return Bounds(min_lat, max_lat, min_lon, max_lon)

    def walk(self, only_points=False):
        for segment_no, segment in enumerate(self.segments):
            for point_no, point in enumerate(segment.points):
                if only_points:
                    yield point
                else:
                    yield point, segment_no, point_no

    def get_points_no(self):
        result = 0

        for track_segment in self.segments:
            result += track_segment.get_points_no()

        return result

    def length_3d(self):
        length = 0
        for track_segment in self.segments:
            d = track_segment.length_3d()
            if d:
                length += d
        return length

    def split(self, track_segment_no, track_point_no):
        """ Splits One of the segments in two parts. If one of the split
        segments is empty it will not be added in the result """
        new_segments = []
        for i in range(len(self.segments)):
            segment = self.segments[i]
            if i == track_segment_no:
                segment_1, segment_2 = segment.split(track_point_no)
                if segment_1:
                    new_segments.append(segment_1)
                if segment_2:
                    new_segments.append(segment_2)
            else:
                new_segments.append(segment)
        self.segments = new_segments

    def join(self, track_segment_no, track_segment_no_2=None):
        """ Joins two segments of this track. If track_segment_no_2 the join will be with the
        next segment """

        if not track_segment_no_2:
            track_segment_no_2 = track_segment_no + 1

        if track_segment_no_2 >= len(self.segments):
            return

        new_segments = []
        for i in range(len(self.segments)):
            segment = self.segments[i]
            if i == track_segment_no:
                second_segment = self.segments[track_segment_no_2]
                segment.join(second_segment)

                new_segments.append(segment)
            elif i == track_segment_no_2:
                # Nothing, it is already joined
                pass
            else:
                new_segments.append(segment)
        self.segments = new_segments

    def get_moving_data(self, stopped_speed_threshold=None):
        moving_time = 0.
        stopped_time = 0.

        moving_distance = 0.
        stopped_distance = 0.

        max_speed = 0.

        for segment in self.segments:
            track_moving_time, track_stopped_time, track_moving_distance, track_stopped_distance, track_max_speed = segment.get_moving_data(stopped_speed_threshold)
            moving_time += track_moving_time
            stopped_time += track_stopped_time
            moving_distance += track_moving_distance
            stopped_distance += track_stopped_distance

            if track_max_speed is not None and track_max_speed > max_speed:
                max_speed = track_max_speed

        return MovingData(moving_time, stopped_time, moving_distance, stopped_distance, max_speed)

    def add_elevation(self, delta):
        for track_segment in self.segments:
            track_segment.add_elevation(delta)

    def add_missing_data(self, get_data_function, add_missing_function):
        for track_segment in self.segments:
            track_segment.add_missing_data(get_data_function, add_missing_function)

    def move(self, location_delta):
        for track_segment in self.segments:
            track_segment.move(location_delta)

    def get_duration(self):
        """ Note returns None if one of track segments hasn't time data """
        if not self.segments:
            return 0

        result = 0
        for track_segment in self.segments:
            duration = track_segment.get_duration()
            if duration or duration == 0:
                result += duration
            elif duration is None:
                return None

        return result

    def get_uphill_downhill(self):
        if not self.segments:
            return UphillDownhill(0, 0)

        uphill = 0
        downhill = 0

        for track_segment in self.segments:
            current_uphill, current_downhill = track_segment.get_uphill_downhill()

            uphill += current_uphill
            downhill += current_downhill

        return UphillDownhill(uphill, downhill)

    def get_location_at(self, time):
        """
        Get locations for this time. There may be more locations because of
        time-overlapping track segments.
        """
        result = []
        for track_segment in self.segments:
            location = track_segment.get_location_at(time)
            if location:
                result.append(location)

        return result

    def get_elevation_extremes(self):
        if not self.segments:
            return MinimumMaximum(None, None)

        elevations = []

        for track_segment in self.segments:
            (_min, _max) = track_segment.get_elevation_extremes()
            if _min is not None:
                elevations.append(_min)
            if _max is not None:
                elevations.append(_max)

        if len(elevations) == 0:
            return MinimumMaximum(None, None)

        return MinimumMaximum(min(elevations), max(elevations))

    def to_xml(self, version=None):
        content = mod_utils.to_xml('name', content=self.name, escape=True)
        content += mod_utils.to_xml('type', content=self.type, escape=True)
        content += mod_utils.to_xml('desc', content=self.description, escape=True)
        if self.number:
            content += mod_utils.to_xml('number', content=self.number)
        for track_segment in self.segments:
            content += track_segment.to_xml(version)

        return mod_utils.to_xml('trk', content=content)

    def get_center(self):
        """ "Average" location for this track """
        if not self.segments:
            return None
        sum_lat = 0
        sum_lon = 0
        n = 0
        for track_segment in self.segments:
            for point in track_segment.points:
                n += 1.
                sum_lat += point.latitude
                sum_lon += point.longitude

        if not n:
            return mod_geo.Location(float(0), float(0))

        return mod_geo.Location(latitude=sum_lat / n, longitude=sum_lon / n)

    def smooth(self, remove_extremes, how_much_to_smooth, min_sameness_distance):
        """ See: GPXTrackSegment.smooth() """
        for track_segment in self.segments:
            track_segment.smooth(remove_extremes, how_much_to_smooth, min_sameness_distance)

    def has_times(self):
        """ See GPXTrackSegment.has_times() """
        if not self.segments:
            return None

        result = True
        for track_segment in self.segments:
            result = result and track_segment.has_times()

        return result

    def has_elevations(self):
        if not self.segments:
            return None

        result = True
        for track_segment in self.segments:
            result = result and track_segment.has_elevations()

        return result

    def get_nearest_location(self, location):
        """ Returns (location, track_segment_no, track_point_no) for nearest location on track """
        if not self.segments:
            return None

        result = None
        distance = None
        result_track_segment_no = None
        result_track_point_no = None

        for i in range(len(self.segments)):
            track_segment = self.segments[i]
            nearest_location, track_point_no = track_segment.get_nearest_location(location)
            nearest_location_distance = None
            if nearest_location:
                nearest_location_distance = nearest_location.distance_2d(location)

            if not distance or nearest_location_distance < distance:
                if nearest_location:
                    distance = nearest_location_distance
                    result = nearest_location
                    result_track_segment_no = i
                    result_track_point_no = track_point_no

        return result, result_track_segment_no, result_track_point_no

    def clone(self):
        return mod_copy.deepcopy(self)





    def __hash__(self):
        return mod_utils.hash_object(self, 'name', 'description', 'number', 'segments')

    def __repr__(self):
        representation = ''
        for attribute in 'name', 'description', 'number':
            value = getattr(self, attribute)
            if value is not None:
                representation += '%s%s=%s' % (', ' if representation else '', attribute, repr(value))
        representation += '%ssegments=%s' % (', ' if representation else '', repr(self.segments))
        return 'GPXTrack(%s)' % representation


class GPXBounds:
    gpx_10_fields = gpx_11_fields = [
            mod_gpxfield.GPXField('min_latitude', attribute='minlat', type=mod_gpxfield.FLOAT_TYPE),
            mod_gpxfield.GPXField('max_latitude', attribute='maxlat', type=mod_gpxfield.FLOAT_TYPE),
            mod_gpxfield.GPXField('min_longitude', attribute='minlon', type=mod_gpxfield.FLOAT_TYPE),
            mod_gpxfield.GPXField('max_longitude', attribute='maxlon', type=mod_gpxfield.FLOAT_TYPE),
    ]

    __slots__ = ('min_latitude', 'max_latitude', 'min_longitude', 'max_longitude')

    def __init__(self, min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None):
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude

    def __iter__(self):
        return (self.min_latitude, self.max_latitude, self.min_longitude, self.max_longitude,).__iter__()

    def __hash__(self):
        return mod_utils.hash_object(self, self.__slots__)

class GPX:
    gpx_10_fields = [
            mod_gpxfield.GPXField('version', attribute=True),
            mod_gpxfield.GPXField('creator', attribute=True),
            mod_gpxfield.GPXField('name'),
            mod_gpxfield.GPXField('description', 'desc'),
            mod_gpxfield.GPXField('author_name', 'author'),
            mod_gpxfield.GPXField('author_email', 'email'),
            mod_gpxfield.GPXField('link', 'url'),
            mod_gpxfield.GPXField('link_text', 'urlname'),
            mod_gpxfield.GPXField('time', type=mod_gpxfield.TIME_TYPE),
            mod_gpxfield.GPXField('keywords'),
            mod_gpxfield.GPXComplexField('bounds', classs=GPXBounds),
            mod_gpxfield.GPXComplexField('waypoints', classs=GPXWaypoint, tag='wpt', is_list=True),
            mod_gpxfield.GPXComplexField('routes', classs=GPXRoute, tag='rte', is_list=True),
            mod_gpxfield.GPXComplexField('tracks', classs=GPXTrack, tag='trk', is_list=True),
    ]
    gpx_11_fields = [
            mod_gpxfield.GPXField('version', attribute=True),
            mod_gpxfield.GPXField('creator', attribute=True),
            'metadata',
                mod_gpxfield.GPXField('name', 'name'),
                mod_gpxfield.GPXField('description', 'desc'),
                'author',
                    mod_gpxfield.GPXField('author_name', 'name'),
                    mod_gpxfield.GPXEmailField('author_email', 'email'),
                    'link',
                        mod_gpxfield.GPXField('author_link', attribute='href'),
                        mod_gpxfield.GPXField('author_link_text', tag='text'),
                        mod_gpxfield.GPXField('author_link_type', tag='type'),
                    '/link',
                '/author',
                'copyright',
                    mod_gpxfield.GPXField('copyright_author', attribute='author'),
                    mod_gpxfield.GPXField('copyright_year', tag='year'),
                    mod_gpxfield.GPXField('copyright_license', tag='license'),
                '/copyright',
                'link',
                    mod_gpxfield.GPXField('link', attribute='href'),
                    mod_gpxfield.GPXField('link_text', tag='text'),
                    mod_gpxfield.GPXField('link_type', tag='type'),
                '/link',
                mod_gpxfield.GPXField('time', type=mod_gpxfield.TIME_TYPE),
                mod_gpxfield.GPXField('keywords'),
                mod_gpxfield.GPXComplexField('bounds', classs=GPXBounds),
                mod_gpxfield.GPXExtensionsField('metadata_extensions', tag='extensions'),
            '/metadata',
            mod_gpxfield.GPXComplexField('waypoints', classs=GPXWaypoint, tag='wpt', is_list=True),
            mod_gpxfield.GPXComplexField('routes', classs=GPXRoute, tag='rte', is_list=True),
            mod_gpxfield.GPXComplexField('tracks', classs=GPXTrack, tag='trk', is_list=True),
            mod_gpxfield.GPXExtensionsField('extensions'),
    ]

    __slots__ = ('version', 'creator', 'name', 'description', 'author_name', 
                 'author_email', 'link', 'link_text', 'time', 'keywords', 
                 'bounds', 'waypoints', 'routes', 'tracks', 'author_link', 
                 'author_link_text', 'author_link_type', 'copyright_author', 
                 'copyright_year', 'copyright_license', 'link_type', 
                 'metadata_extensions', 'extensions')
    
    def __init__(self, tracks=None):
        self.version = None
        self.creator = None
        self.name = None
        self.description = None
        self.link = None
        self.link_text = None
        self.link_type = None
        self.time = None
        self.keywords = None
        self.bounds = None
        self.author_name = None
        self.author_email = None
        self.author_link = None
        self.author_link_text = None
        self.author_link_type = None
        self.copyright_author = None
        self.copyright_year = None
        self.copyright_license = None
        self.metadata_extensions = None
        self.extensions = None
        self.waypoints = []
        self.routes = []
        if tracks:
            self.tracks = tracks
        else:
            self.tracks = []

    def track2trip(self, split_on_new_track=True, split_on_new_track_interval=10, min_sameness_distance=10, min_sameness_interval=2):
        #JOIN
        segment_list = []

        for track in self.tracks:
            if len(track.segments) > 1:
                n_tracks = len(track.segments)
                for seg_no in range(1,len(track.segments)-1):

                    first_segment = track.segments[0].points[-1] #initial segment
                    second_segment = track.segments[seg_no].points[0] #segment in front

                    distance = track.segments[0].points[-1].distance_2d(track.segments[seg_no].points[0]) #to be used as a number of points
                    count = distance / 100.0

                    lat_diff = second_segment.latitude - first_segment.latitude #begins interpolation
                    lon_diff = second_segment.longitude - first_segment.longitude
                    time_diff = first_segment.time - second_segment.time

                    lat_step = lat_diff / (1.0 + int(count))
                    lon_step = lon_diff / (1.0 + int(count))
                    time_step = time_diff.total_seconds() / (1.0 + int(count))

                    time = first_segment.time

                    for i in range(1, 1 + int(count)):
                        sub_lat = first_segment.latitude + (i * lat_step)
                        sub_lon = first_segment.longitude + (i * lon_step)
                        time -= timedelta(seconds=time_step)

                        track.segments[0].points.append(GPXTrackPoint(sub_lat,sub_lon,first_segment.elevation, time)) #ends interpolation #elevation is the same as the first segment. Only latitude, longitude and time are interpolated

                    track.segments[0].points += track.segments[seg_no].points

                track.segments[0].points += track.segments[-1].points


        #SEPARATE

        for track in self.tracks:
            first_point = 1
            if len(track.segments) > 0 and len(track.segments[0].points) > 5 :
                for point_no in range(len(track.segments[0].points[0:-1])):
                    if  track.segments[0].points[point_no+1].time and track.segments[0].points[point_no].time and point_no < len(track.segments[0].points)-1 and track.segments[0].points[point_no+1].time - track.segments[0].points[point_no].time > datetime.timedelta(seconds=split_on_new_track_interval) and \
                        track.segments[0].points[point_no].distance_2d(track.segments[0].points[point_no+1]) < min_sameness_distance:
                        if(point_no != 0):
                            new_segment = track.segments[0].split(first_point, point_no-1)
                            if(len(new_segment.points) > 15 and new_segment.length_2d() > min_sameness_distance and new_segment.points[-1].time - new_segment.points[0].time > timedelta(seconds=split_on_new_track_interval)):
                                segment_list.append(new_segment)
                                first_point += point_no
                                del new_segment

                #print segment_list
                last_segment = track.segments[0].split(first_point, -1)
                if(last_segment.length_2d() > min_sameness_distance and last_segment.points[-1].time - last_segment.points[0].time > timedelta(seconds=split_on_new_track_interval)):
                    segment_list.append(last_segment)
        return segment_list


    def simplify(self, max_distance=None):
        """
        Simplify using the Ramer-Douglas-Peucker algorithm: http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        """
        for track in self.tracks:
            track.simplify(max_distance=max_distance)

    def reduce_points(self, max_points_no=None, min_distance=None):
        """
        Reduce this track to the desired number of points
        max_points = The maximum number of points after the reduction
        min_distance = The minimum distance between two points
        """

        if max_points_no is None and min_distance is None:
            raise ValueError("Either max_point_no or min_distance must be supplied")

        if max_points_no is not None and max_points_no < 2:
            raise ValueError("max_points_no must be greater than or equal to 2")

        points_no = len(list(self.walk()))
        if max_points_no is not None and points_no <= max_points_no:
            # No need to reduce points only if no min_distance is specified:
            if not min_distance:
                return

        length = self.length_3d()

        min_distance = min_distance or 0
        max_points_no = max_points_no or 1000000000

        min_distance = max(min_distance, mod_math.ceil(length / float(max_points_no)))

        for track in self.tracks:
            track.reduce_points(min_distance)

        # TODO
        mod_logging.debug('Track reduced to %s points' % self.get_track_points_no())

    def adjust_time(self, delta):
        """
        Adjust all of the points in all of the segments of all tracks by the given timedelta.
        Adjust with a negative delta in order to subtract time from each point.
        """
        for track in self.tracks:
            track.adjust_time(delta)

    def remove_time(self):
        """ Will remove time metadata. """
        for track in self.tracks:
            track.remove_time()

    def remove_elevation(self, tracks=True, routes=False, waypoints=False):
        """ Will remove elevation metadata. """
        if tracks:
            for track in self.tracks:
                track.remove_elevation()
        if routes:
            for route in self.routes:
                route.remove_elevation()
        if waypoints:
            for waypoint in self.waypoints:
                waypoint.remove_elevation()

    def get_time_bounds(self):
        """
        Returns the first and last found time in the track.
        """
        start_time = None
        end_time = None

        for track in self.tracks:
            track_start_time, track_end_time = track.get_time_bounds()
            if not start_time:
                start_time = track_start_time
            if track_end_time:
                end_time = track_end_time

        return TimeBounds(start_time, end_time)

    def get_bounds(self):
        """
        Get bounds of of this track. Note this method *computes* the bounds i.e. the result may be different
        than the min_latitude, max_latitude, min_longitude and max_longitude properties of this object.
        """
        min_lat = None
        max_lat = None
        min_lon = None
        max_lon = None
        for track in self.tracks:
            bounds = track.get_bounds()

            if not mod_utils.is_numeric(min_lat) or bounds.min_latitude < min_lat:
                min_lat = bounds.min_latitude
            if not mod_utils.is_numeric(max_lat) or bounds.max_latitude > max_lat:
                max_lat = bounds.max_latitude
            if not mod_utils.is_numeric(min_lon) or bounds.min_longitude < min_lon:
                min_lon = bounds.min_longitude
            if not mod_utils.is_numeric(max_lon) or bounds.max_longitude > max_lon:
                max_lon = bounds.max_longitude

        return Bounds(min_lat, max_lat, min_lon, max_lon)

    def get_points_no(self):
        result = 0
        for track in self.tracks:
            result += track.get_points_no()
        return result

    def refresh_bounds(self):
        """
        Compute bounds and reload min_latitude, max_latitude, min_longitude and max_longitude properties
        of this object
        """

        bounds = self.get_bounds()

        self.min_latitude = bounds.min_latitude
        self.max_latitude = bounds.max_latitude
        self.min_longitude = bounds.min_longitude
        self.max_longitude = bounds.max_longitude

    def smooth(self, remove_extremes, how_much_to_smooth):
        """ See GPXTrackSegment.smooth(...) """
        for track in self.tracks:
            track.smooth(remove_extremes, how_much_to_smooth)

    def remove_empty(self):
        """ Removes segments, routes """

        routes = []

        for route in self.routes:
            if len(route.points) > 0:
                routes.append(route)

        self.routes = routes

        for track in self.tracks:
            track.remove_empty()

    def get_moving_data(self, stopped_speed_threshold=None):
        """
        Return a tuple of (moving_time, stopped_time, moving_distance, stopped_distance, max_speed)
        that may be used for detecting the time stopped, and max speed. Not that those values are not
        absolutely true, because the "stopped" or "moving" information aren't saved in the track.

        Because of errors in the GPS recording, it may be good to calculate them on a reduced and
        smoothed version of the track. Something like this:

        cloned_gpx = gpx.clone()
        cloned_gpx.reduce_points(2000, min_distance=10)
        cloned_gpx.smooth(vertical=True, horizontal=True)
        cloned_gpx.smooth(vertical=True, horizontal=False)
        moving_time, stopped_time, moving_distance, stopped_distance, max_speed_ms = cloned_gpx.get_moving_data
        max_speed_kmh = max_speed_ms * 60. ** 2 / 1000.

        Experiment with your own variations to get the values you expect.

        Max speed is in m/s.
        """
        moving_time = 0.
        stopped_time = 0.

        moving_distance = 0.
        stopped_distance = 0.

        max_speed = 0.

        for track in self.tracks:
            track_moving_time, track_stopped_time, track_moving_distance, track_stopped_distance, track_max_speed = track.get_moving_data(stopped_speed_threshold)
            moving_time += track_moving_time
            stopped_time += track_stopped_time
            moving_distance += track_moving_distance
            stopped_distance += track_stopped_distance

            if track_max_speed > max_speed:
                max_speed = track_max_speed

        return MovingData(moving_time, stopped_time, moving_distance, stopped_distance, max_speed)

    def split(self, track_no, track_segment_no, track_point_no):
        track = self.tracks[track_no]

        track.split(track_segment_no=track_segment_no, track_point_no=track_point_no)

    def length_2d(self):
        result = 0
        for track in self.tracks:
            length = track.length_2d()
            if length or length == 0:
                result += length
        return result

    def length_3d(self):
        result = 0
        for track in self.tracks:
            length = track.length_3d()
            if length or length == 0:
                result += length
        return result

    def walk(self, only_points=False):
        """ Use this to iterate through points """
        for track_no, track in enumerate(self.tracks):
            for segment_no, segment in enumerate(track.segments):
                for point_no, point in enumerate(segment.points):
                    if only_points:
                        yield point
                    else:
                        yield point, track_no, segment_no, point_no

    def get_track_points_no(self):
        """ Number of track points, *without* route and waypoints """
        result = 0

        for track in self.tracks:
            for segment in track.segments:
                result += len(segment.points)

        return result

    def get_duration(self):
        """ Note returns None if one of track segments hasn't time data """
        if not self.tracks:
            return 0

        result = 0
        for track in self.tracks:
            duration = track.get_duration()
            if duration or duration == 0:
                result += duration
            elif duration is None:
                return None

        return result

    def get_uphill_downhill(self):
        if not self.tracks:
            return UphillDownhill(0, 0)

        uphill = 0
        downhill = 0

        for track in self.tracks:
            current_uphill, current_downhill = track.get_uphill_downhill()

            uphill += current_uphill
            downhill += current_downhill

        return UphillDownhill(uphill, downhill)

    def get_location_at(self, time):
        """
        Same as GPXTrackSegment.get_location_at(time)
        """
        result = []
        for track in self.tracks:
            locations = track.get_location_at(time)
            for location in locations:
                result.append(location)

        return result

    def get_elevation_extremes(self):
        if not self.tracks:
            return MinimumMaximum(None, None)

        elevations = []

        for track in self.tracks:
            (_min, _max) = track.get_elevation_extremes()
            if _min is not None:
                elevations.append(_min)
            if _max is not None:
                elevations.append(_max)

        if len(elevations) == 0:
            return MinimumMaximum(None, None)

        return MinimumMaximum(min(elevations), max(elevations))

    def get_points_data(self, distance_2d=False):
        """
        Returns a list of tuples containing the actual point, its distance from the start,
        track_no, segment_no, and segment_point_no
        """
        distance_from_start = 0
        previous_point = None

        # (point, distance_from_start) pairs:
        points = []

        for track_no in range(len(self.tracks)):
            track = self.tracks[track_no]
            for segment_no in range(len(track.segments)):
                segment = track.segments[segment_no]
                for point_no in range(len(segment.points)):
                    point = segment.points[point_no]
                    if previous_point and point_no > 0:
                        if distance_2d:
                            distance = point.distance_2d(previous_point)
                        else:
                            distance = point.distance_3d(previous_point)

                        distance_from_start += distance

                    points.append(PointData(point, distance_from_start, track_no, segment_no, point_no))

                    previous_point = point

        return points

    def get_nearest_locations(self, location, threshold_distance=0.01):
        """
        Returns a list of locations of elements like
        consisting of points where the location may be on the track

        threshold_distance is the the minimum distance from the track
        so that the point *may* be counted as to be "on the track".
        For example 0.01 means 1% of the track distance.
        """

        assert location
        assert threshold_distance

        result = []

        points = self.get_points_data()

        if not points:
            return ()

        distance = points[- 1][1]

        threshold = distance * threshold_distance

        min_distance_candidate = None
        distance_from_start_candidate = None
        track_no_candidate = None
        segment_no_candidate = None
        point_no_candidate = None

        for point, distance_from_start, track_no, segment_no, point_no in points:
            distance = location.distance_3d(point)
            if distance < threshold:
                if min_distance_candidate is None or distance < min_distance_candidate:
                    min_distance_candidate = distance
                    distance_from_start_candidate = distance_from_start
                    track_no_candidate = track_no
                    segment_no_candidate = segment_no
                    point_no_candidate = point_no
            else:
                if distance_from_start_candidate is not None:
                    result.append((distance_from_start_candidate, track_no_candidate, segment_no_candidate, point_no_candidate))
                min_distance_candidate = None
                distance_from_start_candidate = None
                track_no_candidate = None
                segment_no_candidate = None
                point_no_candidate = None

        if distance_from_start_candidate is not None:
            result.append(NearestLocationData(distance_from_start_candidate, track_no_candidate, segment_no_candidate, point_no_candidate))

        return result

    def get_nearest_location(self, location):
        """ Returns (location, track_no, track_segment_no, track_point_no) for the
        nearest location on map """
        if not self.tracks:
            return None

        result = None
        distance = None
        result_track_no = None
        result_segment_no = None
        result_point_no = None
        for i in range(len(self.tracks)):
            track = self.tracks[i]
            nearest_location, track_segment_no, track_point_no = track.get_nearest_location(location)
            nearest_location_distance = None
            if nearest_location:
                nearest_location_distance = nearest_location.distance_2d(location)
            if not distance or nearest_location_distance < distance:
                result = nearest_location
                distance = nearest_location_distance
                result_track_no = i
                result_segment_no = track_segment_no
                result_point_no = track_point_no

        return NearestLocationData(result, result_track_no, result_segment_no, result_point_no)

    def add_elevation(self, delta):
        for track in self.tracks:
            track.add_elevation(delta)

    def add_missing_data(self, get_data_function, add_missing_function):
        for track in self.tracks:
            track.add_missing_data(get_data_function, add_missing_function)

    def add_missing_elevations(self):
        def _add(interval, start, end, distances_ratios):
            assert start
            assert end
            assert start.elevation is not None
            assert end.elevation is not None
            assert interval
            assert len(interval) == len(distances_ratios)
            for i in range(len(interval)):
                interval[i].elevation = start.elevation + distances_ratios[i] * (end.elevation - start.elevation)

        self.add_missing_data(get_data_function=lambda point: point.elevation,
                              add_missing_function=_add)

    def add_missing_times(self):
        def _add(interval, start, end, distances_ratios):
            assert start
            assert end
            assert start.time is not None
            assert end.time is not None
            assert interval
            assert len(interval) == len(distances_ratios)

            seconds_between = float(mod_utils.total_seconds(end.time - start.time))

            for i in range(len(interval)):
                point = interval[i]
                ratio = distances_ratios[i]
                point.time = start.time + mod_datetime.timedelta(
                    seconds=ratio * seconds_between)

        self.add_missing_data(get_data_function=lambda point: point.time,
                              add_missing_function=_add)

    def move(self, location_delta):
        for route in self.routes:
            route.move(location_delta)

        for waypoint in self.waypoints:
            waypoint.move(location_delta)

        for track in self.tracks:
            track.move(location_delta)

    def to_xml(self, version=None):
        """
        FIXME: Note, this method will change self.version
        """
        if not version:
            if self.version:
                version = self.version
            else:
                version = '1.0'

        if version != '1.0' and version != '1.1':
            raise GPXException('Invalid version %s' % version)

        self.version = version
        if not self.creator:
            self.creator = 'gpx.py -- https://github.com/tkrajina/gpxpy'

        v = version.replace('.', '/')
        xml_attributes = {
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xmlns': 'http://www.topografix.com/GPX/%s' % v,
                'xsi:schemaLocation': 'http://www.topografix.com/GPX/%s http://www.topografix.com/GPX/%s/gpx.xsd' % (v, v)
        }

        content = mod_gpxfield.gpx_fields_to_xml(self, 'gpx', version, custom_attributes=xml_attributes)

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + content.strip()

    def smooth(self, remove_extremes, how_much_to_smooth, min_sameness_distance, min_sameness_interval):
        for track in self.tracks:
            track.smooth(remove_extremes, how_much_to_smooth, min_sameness_distance, min_sameness_interval)

    def has_times(self):
        """ See GPXTrackSegment.has_times() """
        if not self.tracks:
            return None

        result = True
        for track in self.tracks:
            result = result and track.has_times()

        return result

    def has_elevations(self):
        """ See GPXTrackSegment.has_times() """
        if not self.tracks:
            return None

        result = True
        for track in self.tracks:
            result = result and track.has_elevations()

        return result

    def __hash__(self):
        return mod_utils.hash_object(self, 'time', 'name', 'description', 'author', 'email', 'url', 'urlname', 'keywords', 'waypoints', 'routes', 'tracks', 'min_latitude', 'max_latitude', 'min_longitude', 'max_longitude')

    def __repr__(self):
        representation = ''
        for attribute in 'waypoints', 'routes', 'tracks':
            value = getattr(self, attribute)
            if value:
                representation += '%s%s=%s' % (', ' if representation else '', attribute, repr(value))
        return 'GPX(%s)' % representation

    def clone(self):
        return mod_copy.deepcopy(self)
