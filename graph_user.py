import time, sys, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
from .preprocess import scrape_user

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: graph_user <userDir>")
        exit()

    userDir = sys.argv[1]
    user = userDir.split("/")[-1]
    if not userDir[-1] == "/": userDir += "/"
    js = open(userDir + user + ".json")
    fc = open(userDir + "followCount_proc")
    xfc = []
    yfc = []
    for line in fc.readlines():
        timestamp, followers = line.strip().split(",")
        xfc.append(mdate.epoch2num(int(timestamp)))
        yfc.append(int(followers))
    print("Processed", len(xfc), "follower counts")

    xPost = []
    yPost = []
    instaDir = "/".join(userDir.split("/")[:-2]) + "/"
    posts, _ = scrape_user(instaDir, user, validateMedia = False)
    for post in posts:
        xPost.append(mdate.epoch2num(post['timestamp']))
        xPost.append(post['likes'])

    print("Processed", len(xPost), "posts")

    fig, ax = plt.subplots()
    ax.plot_date(xfc, yfc)
    ax.plot_date(xPos, yPost)

    date_fmt= = '%d-%m-%y'
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.show()

