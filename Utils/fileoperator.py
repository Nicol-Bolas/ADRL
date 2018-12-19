def read_file(filename,strip_last=False):
    mlist = []
    f = open(filename, encoding='utf-8')
    try:
        for line in f.readlines():
            if strip_last:
                mlist.append(line.rstrip('\n'))
            else:
                mlist.append(line)
    finally:
        f.close()
    return mlist