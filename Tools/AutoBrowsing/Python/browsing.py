from selenium import webdriver
import os
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import pandas as pd
import time 

service = Service("/usr/local/bin/geckodriver", log_path="geckodriver.log")
options = Options()
options.headless = True
ssl_keylog_file = "./sslkeylog.log"
os.environ["SSLKEYLOGFILE"] = ssl_keylog_file

profile_path = "./DoH-profile" 
options.add_argument(f"-profile {profile_path}")
options.set_preference("network.trr.mode", 2)  # Enable DoH
options.set_preference("network.trr.uri", "https://dns.nextdns.io/")  # DoH provider

driver = webdriver.Firefox(service=service, options=options)
csv_file_path = "./websites.csv"
weblist = pd.read_csv(csv_file_path)

websites = weblist.iloc[:, 1]
for website in websites :
    driver.get("https://" + website)
    time.sleep(1)

driver.quit()
