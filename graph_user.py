import time, sys, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
from preprocess import scrape_user

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: graph_user <userDir>")
        exit()

    userDir = sys.argv[1].strip("/")
    user = userDir.split("/")[-1]
    instaDir = "/".join(userDir.split("/")[:-1]) + "/"
    userDir += "/"
    js = open(userDir + user + ".json")
    fc = open(userDir + "followCount_proc")
    xfc = []
    fc_timestamps = []
    yfc = []
    for line in fc.readlines():
        timestamp, followers = line.strip().split(",")
        fc_timestamps.append(int(timestamp))
        xfc.append(mdate.epoch2num(int(timestamp)))
        yfc.append(int(followers))
    print("Processed", len(xfc), "follower counts")

    xPost = []
    post_timestamps = []
    yPost = []
    posts, _ = scrape_user(instaDir, user, validateMedia = False)
    for post in posts:
        post_timestamps.append(post['timestamp'])
        xPost.append(mdate.epoch2num(post['timestamp']))
        yPost.append(post['likes'])
    post_timestamps = post_timestamps[::-1]
    xPost = xPost[::-1]
    yPost = yPost[::-1]

    print("Processed", len(xPost), "posts")
    for i in range(len(yPost)):
        if post_timestamps[i] >= fc_timestamps[0]:
            print("breaking at",i)
            xPost = xPost[i:]
            yPost = yPost[i:]
            break

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot_date(xfc, yfc)
    ax2.plot_date(xPost, yPost)

    date_fmt = '%d-%m-%y'
    date_formatter = mdate.DateFormatter(date_fmt)
    ax1.xaxis.set_major_formatter(date_formatter)
    ax2.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.show()

