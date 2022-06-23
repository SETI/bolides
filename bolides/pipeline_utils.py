# get dict containing a list for every feature of the bDispObj
def get_features(bDispObj):

    # get a list with a feature dict for each bolide
    list_of_dicts = [vars(disp.features) for disp in bDispObj.bolideDispositionProfileList]

    # turn it into a dict with a list for every feature
    # assumption: each dict has the same keys
    feature_dict = {key: [dic[key] for dic in list_of_dicts] for key in list_of_dicts[0]}

    return feature_dict
