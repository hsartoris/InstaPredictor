import calendar, datetime
import sys
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ParseFollowCount <followCount>")
        exit()
    f = open(sys.argv[1])
    outDir = sys.argv[1]
    if "/" in outDir:
        # this is shit lol
        while not outDir[-1] == "/":
            outDir = outDir[:-1]

    print("outDir:", outDir)
    line = f.readline().strip()
    f.close()
    data = line.split('\\n" + "')
    data[0] = data[0][1:]
    data[-1] = data[-1][:-3]
    out = np.zeros((len(data), 2), dtype=int)
    for i, datum in enumerate(data):
        date, likes = datum.split(',')
        date = date.split('-')
        stamp = calendar.timegm(datetime.datetime(int(date[0]), int(date[1]), 
                int(date[2]), 12, 0, 0).timetuple())
        out[i] = [stamp, int(likes)]
    np.savetxt(outDir + "followCount_proc", out, delimiter=',', fmt="%i,i")

