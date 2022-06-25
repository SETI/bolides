# get dict containing a list for every feature of the bDispObj
def get_features(bDispObj):

    # get a list with a feature dict for each bolide
    list_of_dicts = [vars(disp.features) for disp in bDispObj.bolideDispositionProfileList]

    # turn it into a dict with a list for every feature
    # assumption: each dict has the same keys
    feature_dict = {key: [dic[key] for dic in list_of_dicts] for key in list_of_dicts[0]}

    return feature_dict


def dict_from_zodb(files, min_confidence):
    import ZODB
    import ZODB.FileStorage
    import zc.zlibstorage
    from tqdm import tqdm

    list_of_dicts = []
    for filename in files:
        bolide_db = ZODB.FileStorage.FileStorage(filename)
        compressed = zc.zlibstorage.ZlibStorage(bolide_db)
        large_record_size = 1 << 27
        db = ZODB.DB(compressed, large_record_size=large_record_size)
        connection = db.open()
        total = connection.root.n_tot_detections
        count = 0
        for key, value in tqdm(connection.root.detections.iteritems(), total=total):
            count += 1
            ID = key
            d = value
            if d.confidence > min_confidence:
                data0 = {'_id': ID, 'confidence': d.confidence,
                         'method': d.confidenceSource, 'comments': d.howFound,
                         'yaw_flip_flag': d.yaw_flip_flag}
                features = vars(d.features)
                stereo_g16 = vars(d.stereoFeatures.G16)
                new_keys = [key+'_g16' for key in stereo_g16.keys()]
                stereo_g16 = dict(zip(new_keys, stereo_g16.values()))
                stereo_g17 = vars(d.stereoFeatures.G16)
                new_keys = [key+'_g17' for key in stereo_g17.keys()]
                stereo_g17 = dict(zip(new_keys, stereo_g17.values()))
                list_of_dicts.append(dict(data0, **features, **stereo_g16, **stereo_g17))
            if count % (total//10) == 0:
                db.cacheMinimize()

    dict_of_lists = {key: [dic[key] for dic in list_of_dicts] for key in list_of_dicts[0]}
    return dict_of_lists
