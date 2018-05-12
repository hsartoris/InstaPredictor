import json, sys, os, shutil
import numpy as np

userFile = "users.txt.bak"


def scrape_user(instaDir, user):
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
        if not os.path.exists(userDir + info['media']):
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
        
    
    

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: preprocess.py <instaDir> <outDir>")
        exit()
    instaDir = sys.argv[1] + "/"
    outDir = sys.argv[2] + "/"
    
    if not os.path.exists(instaDir):
        print("Error: couldn't find instaDir " + instaDir)
        exit()
    if not os.path.exists(instaDir + userFile):
        print("Error: couldn't find user file")
        exit()

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
    for user in users:
        user = user.strip()
        print("Checking user " + user)
        if not os.path.exists(instaDir + user):
            print("Couldn't find dir for " + user + "(" + instaDir + user + ")")
            continue
        user_posts, missed = scrape_user(instaDir, user)
        if user_posts: 
            print("Got", len(user_posts), "for", user + "; missed", missed)
            idx = user_to_data(user, instaDir, user_posts, outDir, ids, idx)
        else:
            print("No posts returned for user " +  user)
    print("Processed", idx, "posts; saved to", outDir)
    ids = np.array(ids).transpose()
    np.savetxt(outDir + "ids", ids, fmt='%i')
        
        
