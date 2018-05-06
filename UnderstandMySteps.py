import gpx_alt as gpxpy
import gpx_alt.gpx

import numpy as np
from scipy.spatial.distance import pdist
from shapely.geometry import Point, LineString
import math
import os
import collections

import codecs
import datetime
import random
import sys
from copy import deepcopy

import MyDict


SimCollection = collections.namedtuple('SimCollection',\
                                       ('indexes', 'ratio', 'names'))

def singleton_file(gpx_set, original=False, seg_set=None):
    '''gpx_set -> set de gpx que serao merged num so ficheiro'''
    new_gpx = gpxpy.gpx.GPX()
    new_gpx.version = '1.1'
    total = 0
    repeated = 0
    for gpx_file in gpx_set:
        duration = gpx_file.description if gpx_file.description is not None else gpx_file.get_duration()
        extension = gpx_file.extensions
        for track in gpx_file.tracks:
            t = track.clone()
            t.description = str(duration)
            if original:
                t.extensions = {'original': True}
            new_gpx.tracks.append(t)
            total += len(t.segments)
            for segment in t.segments:
                if not segment.extensions:
                    break
                segment.extensions['total'] = len(segment.extensions['times'])
                if segment.points[0] == segment.points[-1]:
                    t.segments.remove(segment)
                    repeated += 1
    
    return new_gpx

def remove_extensions(gpx):
    '''Remove todas as extensions de um ficheiro gpx'''
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                point.extensions = None
            segment.extensions = None
        track.extensions = None
    gpx.extensions = None
    return gpx

def bearing(point1, point2):
    """
    Calculates the initial bearing between point1 and point2 relative to north
    (zero degrees).
    
    From: https://www.jonblack.me/how-to-distribute-gps-points-evenly/
    """

    lat1r = math.radians(point1.latitude)
    lat2r = math.radians(point2.latitude)
    dlon = math.radians(point2.longitude - point1.longitude)

    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) \
                        * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(y, x))

def interpolate_distance(points, distance):
    """
    Interpolates points so that the distance between each point is equal
    to `distance` in meters.

    Only latitude and longitude are interpolated; time and elavation are not
    interpolated and should not be relied upon.
    
    From: https://www.jonblack.me/how-to-distribute-gps-points-evenly/
    """

    d = 0
    i = 0
    even_points = []
    while i < len(points):
        if i == 0:
            even_points.append(points[0])
            i += 1
            continue

        if d == 0:
            p1 = even_points[-1]
        else:
            p1 = points[i-1]
        p2 = points[i]

        d += gpxpy.geo.haversine_distance(p1.latitude, p1.longitude,
                                          p2.latitude, p2.longitude)

        if d >= distance:
            brng = bearing(p1, p2)
            ld = gpxpy.geo.LocationDelta(distance=-(d-distance), angle=brng)
            p2_copy = deepcopy(p2)
            p2_copy.move(ld)
            even_points.append(p2_copy)

            d = 0
        else:
            i += 1
    else:
        even_points.append(points[-1])

    return even_points

def find_angle(prev, curr, next):
    a = gpxpy.geo.distance(prev.latitude, prev.longitude, prev.elevation,\
                           curr.latitude, curr.longitude, curr.elevation)
    b = gpxpy.geo.distance(curr.latitude, curr.longitude, curr.elevation,\
                           next.latitude, next.longitude, next.elevation)
    c = gpxpy.geo.distance(prev.latitude, prev.longitude, prev.elevation,\
                           next.latitude, next.longitude, next.elevation)

    cos = min(1, (a*a + b*b - c*c) / (2 * a * b)) if (2 * a * b) != 0 else 0
    cos = max(-1, cos)
    angle = math.acos(cos)
    angle = math.degrees(angle)
    
    return angle, a

def ponto_intersecao(p, r):
    a,b,c = r[0], r[1], r[2]
    x = p.longitude
    y = p.latitude
    p1 = np.array([0, -c/b])
    p2 = np.array([-c/a,0])
    p0 = np.array([x,y])
    P = Point(p0)
    l = LineString([p1,p2])
    
    d = np.linalg.norm(np.cross(p2 - p1, p0 - p1))/np.linalg.norm(p2 - p1)
    n = p2 - p1
    v = p0 - p1
    
    z = p1 + n*(np.dot(v, n)/np.dot(n, n))
    
    return l.distance(P), z

def ponto_intersecao(p, A, B, alt):
    if not alt:
        pB = p.distance_3d(B)
        pA = p.distance_3d(A)
        if pB < 11 or pA < 11: #Esta proximo dos extremos
            if pA < 11 and pB > pA:
                return pA, A, True
            else:
                return pB, B, True
        
    x = p.longitude
    y = p.latitude
    p1 = np.array([A.longitude, A.latitude])
    p2 = np.array([B.longitude,B.latitude])
    p0 = np.array([x,y])

    n = p2 - p1
    v = p0 - p1
    
    dot = np.dot(v, n)
    
    if dot < 0 or dot >= pdist(np.array([p1,p2]), 'sqeuclidean'):
        return 0, None, False
    
    z = p1 + n*(dot/np.dot(n, n))
    
    z = gpxpy.gpx.GPXTrackPoint(longitude=z[0], latitude=z[1])
    
    return p.distance_3d(z), z, False

def unification_segments(s1, s2, alt=False):
    s1_start = s1.points[0]
    s1_end = s1.points[-1]
    
    s2_start = s2.points[0]
    s2_end = s2.points[-1]
    
    pi_start = None
    pi_end = None
    
    dist_a, pi_a, extreme_A = ponto_intersecao(s2_start, s1_start, s1_end, alt)
    dist_b, pi_b, extreme_B = ponto_intersecao(s2_end, s1_start, s1_end, alt)
   
    if dist_a <= 11:
        pi_start = deepcopy(pi_a)
    
    if dist_b <= 11:
        pi_end = deepcopy(pi_b)
        
    return pi_start, pi_end, extreme_A and extreme_B

def merge(gpxs, seg=None):
    def new_seg(pair, segment, track, union={}):
        new_segment = gpxpy.gpx.GPXTrackSegment(points=list(pair))
        new_segment.extensions = {'total': segment.extensions['total'], 'times': segment.extensions['times'].union(union)}
        track.segments.append(new_segment)
        seg_set[pair] |= segment.extensions['times']        
        return
    
    if seg is None:
        seg_set = MyDict.MyDict()
    else:
        seg_set = seg

    size = len(gpxs)

    for i in range(size):
        track = gpxs[i].tracks[0]
        segment_i = 0
        while segment_i < len(track.segments):
            segment = track.segments[segment_i]
            if segment.points[0] == segment.points[-1]:
                track.segments.remove(segment)
                continue
            
            seg_set[(segment.points[0], segment.points[-1])] |= segment.extensions['times']
            segment_deleted = False
            
            for j in range(size):
                if i == j and size != 2:
                    continue
                if segment_deleted:
                    segment_i -= 1
                    break
                alt_track = gpxs[j].tracks[0]
                alt_segment_i = 0
                while alt_segment_i < len(alt_track.segments):
                    alt_segment = alt_track.segments[alt_segment_i]
                    
                    if alt_segment.points[0] == alt_segment.points[-1]:
                        seg_set.pop((alt_segment.points[0], alt_segment.points[-1]), None)
                        alt_track.segments.remove(alt_segment)
                        break
                        
                    if segment.points == alt_segment.points or segment.points == [alt_segment.points[-1], alt_segment.points[0]]:
                        segment.extensions['times'] |= alt_segment.extensions['times']
                        
                        seg_set[(segment.points[0], segment.points[-1])] |= segment.extensions['times']
                        break
                    
                    pi_start, pi_end, extremes = unification_segments(segment, alt_segment)

                    if pi_start == pi_end:
                        alt_segment_i += 1
                        continue

                    original_start = segment.points[0]
                    original_end = segment.points[-1]
                    
                    seg_set[(original_start, original_end)] |= segment.extensions['times']   
                    
                    if extremes:
                        seg_set[(original_start, original_end)] |= segment.extensions['times'].\
                            union(alt_segment.extensions['times']).\
                            union(seg_set.get((alt_segment.points[0], alt_segment.points[-1]),{}))
                        seg_set[(alt_segment.points[0], alt_segment.points[-1])] -= alt_segment.extensions['times']
                        alt_track.segments.remove(alt_segment)
                        break                        
                    
                    to_remove = False
                    to_remove_alt = False
                                        
                    times = alt_segment.extensions['times'].union(seg_set[(alt_segment.points[0], alt_segment.points[-1])])
                    seg_set[(alt_segment.points[0], alt_segment.points[-1])] -= times                   
                    
                    if pi_start is not None:
                        alt_segment.points[0] = deepcopy(pi_start)
                    if pi_end is not None:
                        alt_segment.points[-1] = deepcopy(pi_end)
                        
                    seg_set[(alt_segment.points[0], alt_segment.points[-1])] |= times
                    
                    if pi_start is not None and pi_end is not None:
                        alt_track.segments.remove(alt_segment)
                        seg_set[(alt_segment.points[0], alt_segment.points[-1])] -= times
                        
                        if pi_start != original_start and pi_start != original_end:
                            new_seg((original_start, pi_start), segment, track)
                            to_remove = True
                        elif pi_start == original_end:
                            new_seg((original_start, pi_end), segment, track)
                            to_remove = True                                
                        if pi_end != original_end and pi_end != original_start:
                            new_seg((pi_end, original_end), segment, track)
                            to_remove = True
                        elif pi_end == original_start:
                            new_seg((pi_start, original_end), segment, track)
                            to_remove = True                                 
                        if to_remove:
                            new_seg((pi_start, pi_end), segment, track, times)
                            seg_set[(segment.points[0], segment.points[-1])] -= segment.extensions['times']
                            track.segments.remove(segment)
                            segment_deleted = True
                            break
                        else: #Segmento completo
                            segment.extensions['total'] += total
                            segment.extensions['times'] |= times
                            seg_set[(original_start, original_end)] += times
                    else:
                        alt_segment_i += 1
                        if pi_start is None:
                            
                            if pi_end != original_end and pi_end != original_start:
                                new_seg((original_start, pi_end), segment, track)
                                new_seg((pi_end, original_end), segment, track)
                                seg_set[(original_start, original_end)] -= segment.extensions['times']
                                track.segments.remove(segment)
                                segment_deleted = True  
                                break
                            elif pi_end == original_end:
                                alt_pi_start, alt_pi_end, extremes = unification_segments(alt_segment, segment, True)
                                
                                if alt_pi_start != alt_pi_end and alt_pi_start != segment.points[-1] and alt_pi_start is not None:
                                    seg_set[(original_start, original_end)] -= segment.extensions['times']
                                    segment.points[0] = deepcopy(alt_pi_start)
                                    segment.extensions['times'] |= alt_segment.extensions['times']
                                    seg_set[(segment.points[0], original_end)] |= segment.extensions['times']
                                    
                                    new_seg((alt_segment.points[0], alt_pi_start), alt_segment, alt_track)
                                    new_seg((alt_pi_start, pi_end), alt_segment, alt_track)
                                    
                                    seg_set[(alt_segment.points[0], alt_segment.points[-1])] -= alt_segment.extensions['times']                                    
                                    alt_segment.points[-1] = deepcopy(pi_end)
                                    alt_track.segments.remove(alt_segment)
                                    alt_segment_i -= 1
                                elif alt_pi_start is not None and alt_pi_end is not None:
                                    segment.extensions['times'] |= alt_segment.extensions['times']
                                    seg_set[(segment.points[0], segment.points[-1])] |= alt_segment.extensions['times']
                                
                        elif pi_end is None:
                            if pi_start != original_start and pi_start != original_end:
                                new_seg((original_start, pi_start), segment, track)
                                new_seg((pi_start, original_end), segment, track)
                                seg_set[(original_start, original_end)] -= segment.extensions['times']
                                track.segments.remove(segment)
                                segment_deleted = True         
                                break
                            elif pi_start == original_start:
                                alt_pi_start, alt_pi_end, extremes = unification_segments(alt_segment, segment, alt=True)
                                
                                if alt_pi_start != alt_pi_end and alt_pi_end != segment.points[0] and alt_pi_end is not None:
                                    seg_set[(original_start, original_end)] -= segment.extensions['times']
                                    segment.points[-1] = deepcopy(alt_pi_end)
                                    segment.extensions['times'] |= alt_segment.extensions['times']
                                    seg_set[(original_start, segment.points[-1])] |= segment.extensions['times']
                                    new_seg((pi_start, alt_pi_end), alt_segment, alt_track)
                                    
                                    seg_set[(alt_segment.points[0], alt_segment.points[-1])] -= alt_segment.extensions['times']
                                    alt_segment.points[0] = deepcopy(pi_start)
                                    new_seg((alt_pi_end, alt_segment.points[-1]), alt_segment, alt_track)
                                    alt_track.segments.remove(alt_segment) 
                                    alt_segment_i -= 1
                                elif alt_pi_start is not None and alt_pi_end is not None:
                                    segment.extensions['times'] |= alt_segment.extensions['times']
                                    seg_set[(segment.points[0], segment.points[-1])] |= alt_segment.extensions['times']
                        
            segment_i += 1  
    return seg_set

def transform_track(track):
    segment = track.segments[0]
    ext = segment.extensions
    points = segment.points
    i = 1
    while i < len(points) - 1:
        prev = points[i-1]
        curr = points[i]
        next = points[i+1]
        angle, dist = find_angle(prev, curr, next)
        if 165 <= angle <= 195 and dist <= 11:
            points.remove(points[i])
            continue
        new_segment = gpxpy.gpx.GPXTrackSegment(points=deepcopy(list(segment.points[i-1:i+1])))
        new_segment.extensions = deepcopy(ext)
        track.segments.append(new_segment)
        i += 1
    new_segment = gpxpy.gpx.GPXTrackSegment(points=list(segment.points[i-1:i+1]))
    new_segment.extensions = deepcopy(ext)
    track.segments.append(new_segment)       
    track.segments.remove(segment)
    
def parse_folder(addr, real):
    '''
    addr -> path to folder with gpx files
    Creates .txt file in addr with names of files
    Returns a list with gpxpy.GPX instances smoothed and simplified
    '''
    files = os.listdir(addr)
    gpxs = []
    raw = []
    i = 1
    singles = dict()
    txt_files = codecs.open(os.path.join(addr, 'mods', 'files.txt'), 'w', 'utf-8')
    min_t = datetime.datetime.today()
    max_t = datetime.datetime.min    

    for file in files:
        if '.gpx' not in file or file == 'final.gpx' or file == 'avg.gpx':
            continue
        
        txt_files.write('mod_' + file + '\n')
        
        print("Doing " + file)
        file_name = os.path.join(addr, file)
        f = codecs.open(file_name, 'r', 'utf-8')
        gpx = gpxpy.parse(f)
        f.close()
        
        gpx = remove_extensions(gpx)
        
        times = set([gpx.get_time_bounds().start_time])

        time = max(times)            
        
        single_gpx = gpxpy.gpx.GPX()
        single_gpx.version = '1.1'
        single_gpx.name = time.date().isoformat() + '_' + str(i)    
        
        gpx_write = single_gpx.clone()
        gpx_write.version = '1.1'
        
        single_gpx.tracks.append(gpxpy.gpx.GPXTrack())
        for t in times:
            if t < min_t:
                min_t = t
            if t > max_t:
                max_t = t
            singles[deepcopy(t)] = single_gpx.clone()
        i += 1
        gpx.remove_elevation()
        raw += [deepcopy(gpx)]        
        dur = gpx.get_duration()
        leng = gpx.length_3d()
        t1 = gpx.tracks[0].segments[0].points[0].time
        t2 = gpx.tracks[0].segments[0].points[1].time
        time = (t2-t1).seconds
        eps = (leng/dur) * time

        gpx_track = gpxpy.gpx.GPXTrack()
        
        for s in gpx.tracks[0].segments:
            s.simplify3()
            
            if real:
                s.points = interpolate_distance(s.points, 11)
                s.simplify(min(11,round(eps+1)))
            
            # Create first segment in our GPX track:
            s.extensions= {'total': 1, 'times': set(times)}
            gpx_track.segments.append(s)
        
        transform_track(gpx_track)
        gpx_track.extensions = {'original': True}
        gpx_write.tracks.append(gpx_track)
        gpx_write.remove_time()
        gpxs.append(gpx_write)
        to_write = gpx_write.clone()
        for s in to_write.tracks[0].segments:
            times = set()
            for t in s.extensions['times']:
                times.add(t.isoformat())
            s.extensions['times'] = times
        f = open(os.path.join(addr, 'mods', 'mod_' + file), 'w')
        f.write(to_write.to_xml())
        f.close()
    
    txt_files.write(min_t.isoformat() + '\n')
    txt_files.write(max_t.isoformat() + '\n')
    txt_files.close()
    return gpxs, raw, singles, min_t, max_t

def complete_file(gpxs, avg):
    gpx = singleton_file(gpxs, original=True)
    
    for track in avg.tracks:
        gpx.tracks.append(track)
        
    return gpx

def set_to_gpx(set_seg, singles):
    now = datetime.datetime.now().date()
    gpx = gpxpy.gpx.GPX()
    gpx.version = '1.1'
    track = gpxpy.gpx.GPXTrack()
    max_count = 0
    min_count = sys.maxsize
    for seg in set_seg: 
        if len(set_seg[seg]) == 0: #Empty set
            continue
        segment = gpxpy.gpx.GPXTrackSegment()
        p = tuple(seg)
        segment.points.append(p[0])
        segment.points.append(p[-1])
        total = len(set_seg[seg])
        
        #Calculate description, extensions and singles
        ext = list()
        s = datetime.timedelta()
        for t in sorted(set_seg[seg]):
            ext.append(t.isoformat())
            s += (now - t.date())
            singles[t].tracks[0].segments.append(segment)    
        segment.description = str(now - (s // total))
        segment.extensions = {'times': ext, 'total': total}

        if total > max_count:
            max_count = total
        if total < min_count:
            min_count = total
                
        track.segments.append(segment)
    gpx.tracks.append(track)
    return gpx, max_count, min_count

def create_singles(singles, addr, max_count, min_count, min_t, max_t):
    path = os.path.join(addr, 'singles')
    f = open(os.path.join(path, 'list.txt'), 'w')
    for t in singles:
        file = open(os.path.join(path, singles[t].name + '.gpx'), 'w')
        file.write(singles[t].to_xml())
        file.close()
        f.write(singles[t].name + '\n')
    f.write(min_t.isoformat() + '\n')
    f.write(max_t.isoformat() + '\n')
    f.write(str(max_count) + '\n')    
    f.close()
    txt_files = open(os.path.join(addr, 'mods', 'files.txt'), 'a')
    txt_files.write(str(max_count) + '\n')
    txt_files.write(str(min_count) + '\n')
    txt_files.close()
    return

def raw_view(addr):
    gpxs = []  
    files = os.listdir(addr)
    for file in files:
        if '.gpx' not in file or file == 'final.gpx' or file == 'avg.gpx':
            continue
                
        print("Doing " + file)
        file_name = os.path.join(addr, file)
        f = codecs.open(file_name, 'r', 'utf-8')
        gpx = gpxpy.parse(f)
        f.close()
        gpx = remove_extensions(gpx)
        gpxs.append(gpx)
    file = open(os.path.join(addr, 'avg.gpx'), 'w')
    file.write(singleton_file(gpxs).to_xml())
    file.close()    

def main(addr, complete=False, real=True):
    gpxs, raw_gpxs, singles, min_t, max_t = parse_folder(addr, real)
    #print("Imports done. Imported", len(gpxs))

    seg_set = merge(gpxs)
    avg, max_count, min_count = set_to_gpx(seg_set, singles)
    create_singles(singles, addr, max_count, min_count, min_t, max_t)
    file = open(os.path.join(addr, 'avg.gpx'), 'w')
    file.write(avg.to_xml())
    file.close()
    if complete:
        final_gpx = complete_file(raw_gpxs, avg)
        file = open(os.path.join(addr, 'final.gpx'), 'w')
        file.write(final_gpx.to_xml())
        file.close()  
        
    return avg