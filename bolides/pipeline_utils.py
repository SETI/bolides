def dict_from_zodb(files, min_confidence):
    import ZODB
    import ZODB.FileStorage
    import zc.zlibstorage
    from tqdm import tqdm

    list_of_dicts = []
    for filename in files:

        # use ZODB and zc to open the compressed database
        bolide_db = ZODB.FileStorage.FileStorage(filename)
        compressed = zc.zlibstorage.ZlibStorage(bolide_db)
        large_record_size = 1 << 27
        db = ZODB.DB(compressed, large_record_size=large_record_size)
        connection = db.open()

        # get number of detections
        total = connection.root.n_tot_detections

        # create an iterator that goes through all of the detections
        iterator = connection.root.detections.iteritems(),

        count = 0
        # loop through iterator, with loading bar
        for key, value in tqdm(iterator, total=total):
            count += 1
            ID = key
            d = value

            # only add to list of dicts if the confidence is above the threshold
            if d.confidence > min_confidence:
                # create dict containing basic data
                data0 = {'_id': ID, 'confidence': d.confidence,
                         'method': d.confidenceSource, 'comments': d.howFound,
                         'yaw_flip_flag': d.yaw_flip_flag}
                # get dict from the attributes and values of the features object
                features = vars(d.features)
                # get dicts from the attributes and values of the stereo features objects
                stereo_g16 = add_key_suffix(vars(d.stereoFeatures.G16), '_g16')
                stereo_g17 = add_key_suffix(vars(d.stereoFeatures.G17), '_g17')

                # append a dict combining all of these to the list of dicts
                list_of_dicts.append(dict(data0, **features, **stereo_g16, **stereo_g17))

            # minimize the cache a few times while running
            if count % (total//10) == 0:
                db.cacheMinimize()

    # turn the list of dicts into a dict of lists
    # (under the assumption that all dicts have the same keys)
    dict_of_lists = {key: [dic[key] for dic in list_of_dicts] for key in list_of_dicts[0]}

    return dict_of_lists


# helper function to add a suffix to each key of a dict
def add_key_suffix(dic, suffix):
    new_keys = [key+suffix for key in dic.keys()]
    return dict(zip(new_keys, dic.values()))
