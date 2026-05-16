targets = ['nas', 'eminem', 'dr. dre', '2pac', 'big l', 'notorious b.i.g.', 'outkast', 'ice cube', 'jay-z', 'snoop dogg', 'wu-tang clan', 'a tribe called quest']
with open('Data/audioscrobbler/artist_data.txt', 'rb') as f:
    for line in f:
        try:
            line = line.decode('utf-8', errors='ignore').strip()
            parts = line.split('\t')
            if len(parts) == 2:
                aid, name = parts
                if name.lower().strip() in targets:
                    print(f'{aid}\t{name}')
        except:
            pass
