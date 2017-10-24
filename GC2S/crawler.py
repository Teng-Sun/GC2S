
# coding: utf-8
import pickle
#sqlalchemy.__version__
from sqlalchemy import create_engine
engine = create_engine('sqlite:///HR_new.db', echo=True)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy import Column, Integer, String, Float, ARRAY
from IPython import embed
def tablerepr(self):
    return "<{}({})>".format(
        self.__class__.__name__,
        ', '.join(
            ["{}={}".format(k, repr(self.__dict__[k]))
                for k in sorted(self.__dict__.keys())
                if k[0] != '_']
        )
    )
Base.__repr__ = tablerepr

class HR(Base):
    __tablename__ = 'hotel-review'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    h_score = Column(Float)
    u_score = Column(Float)
    staff_score = Column(Float)
    clean_score = Column(Float)
    loc_score = Column(Float)
    wifi_score = Column(Float)
    equip_score = Column(Float)
    comfort_score =Column(Float)
    cp_score = Column(Float)
    review = Column(String)
Base.metadata.create_all(engine)
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
Session.configure(bind=engine)
sess = Session()

import requests
import sqlalchemy
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import re
zh_words = re.compile(u"[\u4e00-\u9fa5]+")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
TIMEOUT = 20

def get_comment(name, num, polarity, id_):
    s = get_attri(name, id_)
    #u_score = score
    h_score = s[0]
    staff_score = s[1]
    clean_score = s[2]
    loc_score = s[3]
    wifi_score = s[4]
    equip_score = s[5]
    comfort_score = s[6]
    cp_score = s[7]
    for j in range(6):
        num = j * 10
        if polarity :
            link = "https://www.booking.com/reviewlist.zh-tw.html?pagename="+str(name)+";cc1=tw;rows=10;offset="+str(num)
        else:
            link = "https://www.booking.com/reviewlist.zh-tw.html?pagename="+str(name)+";score=review_adj_poor;cc1=tw;rows=10;offset="+str(num)
        raw_data = requests.get(link, headers=headers, timeout=TIMEOUT).text
        soup = BeautifulSoup(raw_data, "lxml")
        #print(soup)
        #embed()
        review_list = soup.find_all("li",class_="review_item clearfix ")
        #print('review list -->', review_list)
        for rl in review_list:
            s = rl.find("span",class_="review-score-badge")
            score = float(s.string.replace('\n', ''))
            if score > 0.6 :
                try:
                    tmp = rl.find("p", class_="review_pos")
                except:
                    break
            else:
                try:
                    tmp = rl.find("p", class_="review_neg")
                except:
                    break
            if tmp != None:
                c = [n for n in re.findall(u"[\u4e00-\u9fff\d]+", tmp.prettify())]
                review = ",".join(c)
                print(review)
                u_score = score
                one = HR(name=name,
                        u_score=u_score,
                        h_score=h_score,
                        staff_score=staff_score,
                        clean_score=clean_score,
                        loc_score=loc_score,
                        wifi_score=wifi_score,
                        equip_score=equip_score,
                        comfort_score=comfort_score,
                        cp_score=cp_score,
                        review=review)
                #print(one)
                sess.add(one)
                sess.commit()

def get_hotel(city, num):
	link = "https://www.booking.com/searchresults.zh-tw.html?city="+city+"&ssb=empty&sr_ajax=1&rows=10&offset="+str(num)
	raw_data = requests.get(link, headers=headers, timeout=TIMEOUT).text
	soup = BeautifulSoup(raw_data, "lxml")
	data = soup.find_all("a", class_="hotel_name_link url")
	data = [ i["href"].split(".zh-tw.html?")[0].split("/")[-1] for i in data ]
	return data

def get_attri(hotel, id_) :
    col = {"員工素質":1, "整潔度":2, "住宿地點":3, "免費 WiFi":4 ,"設施":5, "舒適程度":6, "性價比":7 }
    link = 'https://www.booking.com/hotel/tw/'+hotel+'.zh-tw.html?dest_type=city;dest_id='+str(id_)
    raw_data = requests.get(link, headers=headers, timeout=TIMEOUT).text
    soup = BeautifulSoup(raw_data, "lxml")
    data = soup.find("div", class_=" review_list_score_container lang_ltr")
    s = data.find("span", class_="review-score-badge")
    ans = [float(s.string), 0, 0, 0, 0, 0, 0, 0]
    data = data.find("span", class_=" review_list_score_breakdown_col")
    #print(data)
    dd = data.find_all("li", class_="clearfix one_col")
    #print(dd)
    for item in dd:
        sn = item.find("p", class_="review_score_name").string
        #print(sn)
        if sn in col:
            ans[col[sn]] = float(item.find("p", class_="review_score_value").string)
    return ans
    exit()
if __name__== '__main__':
    err = []
                #高雄           台北        台中
    for city in ['-2632378', '-2637882', '-2637824'] :
        for i in range(60):
            data = get_hotel(city, 10*i)
            for hotel in data :
                sleep(0.2)
                print(hotel)
                try:
                    get_comment(hotel, 60, 1, city)
                except:
                    err.append(hotel)

    with open('err', 'wb') as f:
        pickle.dump(err, f)

