import json, sys, os, shutil, datetime, calendar, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np

userFile = "users.txt.bak"

class Modes():
    TRAININGDATA = 0
    ANALYZE = 1

def scrape_user(instaDir, user, validateMedia=True):
    posts = []
    userDir = instaDir + user + "/"
    data = None
    missed = 0
    try: 
        jsonf = open(userDir + user + ".json")
        data = json.load(jsonf)
    except IOError:
        return None, 0
    for post in data:
        info = dict()
        info['likes'] = post['edge_media_preview_like']['count']
        info['media'] = post['display_url'].split("/")[-1]
        info['timestamp'] = int(post['taken_at_timestamp'])
        if not os.path.exists(userDir + info['media']) and validateMedia:
            print("Couldn't find post media: " + info['media'])
            missed += 1
            continue
        # this needs debugging for individual image posts
        #caption = post['edge_media_to_caption']['edges'][0]['node']['text']
        #caption = str(caption.encode('utf-8')).replace('\n',' ')
        #caption = caption.replace("\'", "'")
        #info['caption'] = caption
        #print(info)
        posts.append(info)
    return posts, missed

def user_to_data(user, instaDir, user_posts, outDir, ids, idx=0):
    userDir = instaDir + user + "/"
    for post in user_posts:
        shutil.copyfile(userDir + post['media'], outDir + str(idx) + ".jpg")
        ids.append(post['likes'])
        idx += 1
    return idx

def analyze2(posts):
    # monday is 0
    fig, axes = plt.subplots(1,7)
    user_likes = []
    yMax = 0
    for (user, user_posts) in posts:
        if user == "actuallyaxley" or user == "vmalik7": continue
        days = []
        for i in range(7): days.append([[],[]])
        for post in user_posts:
            days[time.localtime(post['timestamp'])[6]][0].append(
                time.localtime(post['timestamp'])[3])
            days[time.localtime(post['timestamp'])[6]][1].append(
                post['likes'])
            if post['likes'] > yMax:
                yMax = post['likes']
        user_likes.append(days)
        for i in range(len(days)):
            line, = axes[i].plot(user_likes[-1][i][0], user_likes[-1][i][1], 
                    'o', label=user)

    yMax += 50
    for ax in axes:
        ax.set_ylim([0, yMax])
    axes[0].legend()
    plt.show()

def analyze(posts):
    # monday is 0
    fig, axes = plt.subplots(1,7)
    hours = [i for i in range(24)]
    user_likes = []
    yMax = 0
    for (user, user_posts) in posts:
        if not user == "betthole": continue
        days = []
        for i in range(7): days.append([0]*24)
        print(days)
        for post in user_posts:
            day = time.localtime(post['timestamp'])[6]
            hour = time.localtime(post['timestamp'])[3]
            
            days[day][hour] += post['likes']
        if np.max(days) > yMax:
            yMax = np.max(days)
        user_likes.append(days)
        print(hours)
        print(days[0])
        for i in range(len(days)):
            line, = axes[i].plot(hours, user_likes[-1][i], label=user)

    yMax += 50
    for ax in axes:
        ax.set_ylim([0, yMax])
    axes[0].legend()
    plt.show()

if __name__ == "__main__":
    mode = Modes.ANALYZE
    if mode == Modes.TRAININGDATA and not len(sys.argv) == 3:
        print("Usage: preprocess.py <instaDir> <outDir>")
        exit()
    elif mode == Modes.TRAININGDATA:
        outDir = sys.argv[2] + "/"
    elif mode == Modes.ANALYZE and not len(sys.argv) == 2:
        print("Usage: preprocess.py <instaDir>")
        exit()

    instaDir = sys.argv[1] + "/"
    
    if not os.path.exists(instaDir):
        print("Error: couldn't find instaDir " + instaDir)
        exit()
    if not os.path.exists(instaDir + userFile):
        print("Error: couldn't find user file")
        exit()

    if mode == Modes.TRAININGDATA:
        if not os.path.exists(outDir):
            print("Creating output directory: " + outDir)
            os.mkdir(outDir)
        if not os.path.exists(outDir + "images"):
            os.mkdir(outDir + "images")
    
    f = open(instaDir + userFile)
    users = f.readlines()
    f.close()
    
    ids = []
    idx = 0
    posts = []
    for user in users:
        user = user.strip()
        print("Checking user " + user)
        if not os.path.exists(instaDir + user):
            print("Couldn't find dir for " + user + "(" + instaDir + user + ")")
            continue
        user_posts, missed = scrape_user(instaDir, user,
                validateMedia=(mode==Modes.TRAININGDATA))
        if user_posts: 
            print("Got", len(user_posts), "for", user + "; missed", missed)
            if mode == Modes.TRAININGDATA:
                idx = user_to_data(user, instaDir, user_posts, outDir, ids, idx)
            posts.append((user,user_posts))
        else:
            print("No posts returned for user " +  user)
            continue

    if mode == Modes.TRAININGDATA:
        print("Processed", idx, "posts; saved to", outDir)
        ids = np.array(ids).transpose()
        np.savetxt(outDir + "ids", ids, fmt='%i')
    elif mode == Modes.ANALYZE:
        analyze2(posts)

        
        
