import time, sys, matplotlib.date, datetime
from .preprocess import scrape_user

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: graph_user <userDir>")
        exit()

    userDir = sys.argv[1]
    user = userDir.split("/")[-1]
    if not userDir[-1] == "/": userDir += "/"
    js = open(userDir + 
