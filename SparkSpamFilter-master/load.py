from pymongo import MongoClient
import codecs
import curses

def connectToMongo():
    try:
        client = MongoClient('localhost',27017)
        db = client.SparkSpamFilter
        print("connected to MongoDb")
        docCount = db.emails.count_documents({})
        if( docCount != 0):
            print("email collection is full and contains "+str(docCount)+" documents!!!")
            print("dropping the email collection...")
            db.emails.drop()
            print("email collection dropped")
        
        return db
    except IndexError:
        print("Couldn't connect to the MongoDb")


def readDataset(filename):
    data = codecs.open(filename, 'r', encoding='utf-8')
    return data


def fillDb(data,db):
    index = 0
    stdscr = curses.initscr()
    for line in data:
        temp = line.rstrip().split('\t')
        record = {
            "class": temp[0],
            "text": temp[1]
        }
        index+=1
        stdscr.addstr(0, 0,"inserted "+str(index) +" docuemnts...")
        stdscr.refresh()
        db.emails.insert_one(record)



db = connectToMongo()
data = readDataset("data/emailCollection")
fillDb(data,db)






