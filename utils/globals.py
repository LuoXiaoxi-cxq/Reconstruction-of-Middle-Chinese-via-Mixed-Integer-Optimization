def make_dict(ls):
    dct = dict()
    for i, content in enumerate(ls):
        dct[content] = i
    return dct


PLACE_LS = ['北京', '济南', '西安', '太原', '武汉', '成都', '合肥', '扬州', '苏州', '温州',
            '长沙', '双峰', '南昌', '梅县', '广州', '阳江', '厦门', '潮州', '福州', '建瓯']

GY_INITIAL = ['幫', '滂', '並', '明', '端', '透', '定', '泥', '知', '徹', '澄', '孃',
              '精', '清', '從', '心', '邪', '莊', '初', '崇', '生', '俟', '章', '昌', '船', '書', '禪',
              '見', '溪', '群', '疑', '影', '以', '云', '曉', '匣', '來', '日']

INITIAL_DICT = make_dict(GY_INITIAL)

FEATURE_LS = ['continuant', 'delayed_release', 'sonority', 'voice', 'spread_gl', 'LABIAL', 'labiodental', 'CORONAL',
              'anterior', 'distributed', 'lateral', 'DORSAL', 'high', 'front']

# independent feature
I_FEATURE_LS = ['continuant', 'sonority', 'voice', 'spread_gl', 'LABIAL', 'CORONAL',
                'lateral', 'DORSAL', 'nasal', 'round']

# dependent feature
D_FEATURE_DICT = {'delayed_release': 'sonority', 'labiodental': 'LABIAL', 'anterior': 'CORONAL',
                  'distributed': 'CORONAL', 'high': 'DORSAL', 'front': 'DORSAL'}
SCALE_DICT = {'high': 3, 'front': 3, 'sonority': 5}
