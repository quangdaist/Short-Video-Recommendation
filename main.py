import random
import string
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import tkinter as tk
# from selenium.webdriver.common.keys import Keys
# from selenium_stealth import stealth
import time
import os
import sys
from datetime import datetime
import getpass
# import subprocess

options = webdriver.ChromeOptions()

options.add_argument("start-maximized")
options.add_argument('--no-sandbox')
options.add_argument("--allow-running-insecure-content")
options.add_argument("--disable-popup-blocking")
options.add_argument("no-default-browser-check")
options.add_argument("--profile-directory=Default")
direc_usr_data = f"{os.getcwd()}\\User Data"
options.add_argument(f"--user-data-dir={direc_usr_data}")
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-dev-shm-usage')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# driver_path = 'https://github.com/DaiVo20/Short-Video-Recommendation/blob/efe162bd8e012c1017abda15a64efc47b30a157b/chromedriver.exe'
driver = webdriver.Chrome(options=options, executable_path='chromedriver.exe')

URL = "https://www.tiktok.com/foryou?is_copy_url=1&is_from_webapp=v1"
driver.get(URL)

# only pause the script in the first time
time.sleep(3)
try:
    _ = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div/div[2]/button')
except:
    pass
else:
    input('Press Enter after you\'ve logged in')

# create action chain object
action = ActionChains(driver)
first_vid_ele = driver.find_element(By.XPATH,
                                    '//*[@id="app"]/div[2]/div[2]/div[1]/div[1]/div/div[2]/div[1]/div/div[1]/div')
action.move_to_element(first_vid_ele).click().perform()


def is_logged_in():
    try:
        logged_in = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div/div[2]/div[4]')
        logged_in_button.config(text='Logged-in', fg="black", bg='green')
    except:
        pass


def press_button(button):
    global time_swift
    global like
    global action
    global driver

    flag = False
    if button == 'Up':
        ele = '//*[@id="app"]/div[2]/div[3]/div[1]/button[2]'
    elif button == 'Down':
        ele = '//*[@id="app"]/div[2]/div[3]/div[1]/button[3]'
    elif button == 'Like':
        ele = '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[1]/div[1]/button[1]'
        like = 1
        flag = True

    try:
        if not flag:
            time_swift = time.time()
            record_history()
            like = 0
        button_ele = driver.find_element(By.XPATH, ele)
        action.move_to_element(button_ele).click().perform()
    except:
        pass



def record_history():
    global time_swift
    global history
    global driver
    global action
    global like
    url_vid = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[2]/p').text
    like_count = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[1]/div['
                                               '1]/button[1]/strong').text
    desc_vid = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[1]').text
    cmt_count = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[1]/div['
                                              '1]/button[2]/strong').text
    ele_time_container = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[1]/div[2]/div[2]')
    # like = driver.find_element(By.XPATH,
    #                            '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[1]/div[1]/button[1]/span/svg')
    action.move_to_element(ele_time_container).perform()
    time_container = ele_time_container.text

    while not time_container:
        ele_time_container = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[1]/div[2]/div[2]')
        time_container = ele_time_container.text

    results = [url_vid, desc_vid, like_count, cmt_count, like, time_container, time_swift]

    history.append(results)
    print([url_vid[40:61], desc_vid[:10], like_count, cmt_count, like, time_container, time_swift])


def get_random_string(length):
    global now
    global time_swift
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_ = ''.join(random.choice(letters) for i in range(length))
    result_str = f'{time_swift}_{now.timestamp()}_{result_}'
    return result_str


def display_history():
    global history
    cols = ['url', 'desc_video', 'like_count', 'comment_count', 'like', 'time_container', 'timestamp']
    df = pd.DataFrame(history, columns=cols)
    if not df.empty:
        history = list()
        user_name = getpass.getuser()
        df['user'] = [user_name] * len(df)
        file_name = get_random_string(20)
        is_exist_raw = os.path.exists('./raw')
        if not is_exist_raw:
            os.mkdir('./raw')
        file_path = f"raw/{file_name}.csv"
        df.to_csv(file_path, index=False)
        print(df)


now = datetime.now()
time_now = now.strftime("%H:%M:%S")
# history of user when suffering TikTok
time_swift = time.time()
history = list()
like = 0

# Tkinter
app = tk.Tk()
app.geometry('250x400')
app.wm_title("TikTok label")


logged_in_button = tk.Button(app, width=25, height=5, text="Log in ?", bg="white",
                             fg='red', command=lambda: is_logged_in())
logged_in_button.pack(expand=True)

up_button = tk.Button(app, width=10, height=5, text="Up (^)", fg="black",
                      command=lambda: press_button('Up'))
up_button.pack(expand=True)

down_button = tk.Button(app, width=10, height=5, text="Down (v)", fg="green",
                        command=lambda: press_button('Down'))
down_button.pack(expand=True)

like_button = tk.Button(app, width=25, height=5, text="Like (l)", fg="black", bg='red',
                        command=lambda: press_button('Like'))
like_button.pack(expand=True)

view_file = tk.Button(app, width=15, height=5, text="Display & Save \nhistory", fg="green",
                      command=lambda: display_history())
view_file.pack(side='left', expand=True)

keyboard_control = {'Up': 'Up', 'l': 'Like', 'L': 'Like', 'Down': 'Down'}


def show_keyboard(event):
    key_ = event.keysym
    # print("You pressed: " + key_)
    keyboard.config(text=key_)
    if key_ in keyboard_control.keys():
        press_button(keyboard_control[key_])


app.bind("<Key>", show_keyboard)
keyboard = tk.Label(app, width=15, height=5, font=("Helvetica", 20))
keyboard.pack(side='right')

app.mainloop()
