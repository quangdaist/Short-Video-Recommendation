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
from datetime import datetime

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver_path = r"D:\OneDrive - Trường ĐH CNTT - University of Information Technology\Máy tính\app_tracking\chromedriver.exe"
driver = webdriver.Chrome(options=options, executable_path=driver_path)
# stealth(driver,
#         languages=["en-US", "en"],
#         vendor="Google Inc.",
#         platform="Win32",
#         webgl_vendor="Intel Inc.",
#         renderer="Intel Iris OpenGL Engine",
#         fix_hairline=True,
#         )

URL = "https://www.tiktok.com/foryou?is_copy_url=1&is_from_webapp=v1"
driver.get(URL)
time.sleep(3)
# input('Press Enter after you\'ve logged in')
soup = BeautifulSoup(driver.page_source, 'html.parser')
feed_video = soup.find('div', {'data-e2e': 'feed-video'})

for vid in feed_video:
    item = vid.find('div', {'mode': "0"})
    if item is not None:
        ite = item.contents[1]
        first_vid_x_path = ite.contents[0].attrs['id']

# create action chain object
action = ActionChains(driver)
first_vid_ele = driver.find_element(By.XPATH, f'//*[@id="{first_vid_x_path}"]/video')
action.move_to_element(first_vid_ele).click().perform()


def update_time():
    _now = datetime.now()
    _time_now = _now.strftime("%H:%M:%S")
    time_clock.config(text=f"{_time_now}")
    app.after(1000, update_time)


def press_button(button):
    global time_swift
    global like
    global action

    ele = None
    flag = False
    if button == 'Up':
        ele = '//*[@id="app"]/div[2]/div[3]/div[1]/button[2]'
    elif button == 'Down':
        ele = '//*[@id="app"]/div[2]/div[3]/div[1]/button[3]'
    elif button == 'Like':
        # ele = '//*[@id="app"]/div[2]/div[3]/div[2]/div[2]/div[2]/div[1]/div[1]/button[1]'
        like = 1
        flag = True
    old_time = time.time()

    if ele is not None:
        button_ele = driver.find_element(By.XPATH, ele)
        action.move_to_element(button_ele).click().perform()
        time_swift = old_time
        record_history()
        like = 0

    # try:
    #     button_ele = driver.find_element(By.XPATH, ele)
    #     action.move_to_element(button_ele).click().perform()
    #     if not flag:
    #         time_swift = old_time
    #         record_history()
    #         like = 0
    #
    # except:
    #     print("Input invalid!!!")


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
    action.move_to_element(ele_time_container).click().perform()
    time_container = ele_time_container.text

    while not time_container:
        ele_time_container = driver.find_element(By.XPATH, '//*[@id="app"]/div[2]/div[3]/div[1]/div[2]/div[2]')
        time_container = ele_time_container.text

    results = [url_vid, desc_vid, like_count, cmt_count, like, time_container, time_swift]

    history.append(results)
    # print(results)
    print(time_container)
    # return results


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
        file_name = get_random_string(20)
        df.to_csv(f'{file_name}.csv', index=False)
        print(df)


def upload2drive():
    return None


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

time_clock = tk.Label(app, text=f"{time_now}")
time_clock.pack(expand=True)
app.after(1000, update_time)

up_button = tk.Button(app, width=25, height=5, text="Up", fg="black",
                      command=lambda: press_button('Up'))
up_button.pack(expand=True)

like_button = tk.Button(app, width=25, height=5, text="Like", fg="black", bg='red',
                        command=lambda: press_button('Like'))
like_button.pack(expand=True)

down_button = tk.Button(app, width=25, height=5, text="Down", fg="green",
                        command=lambda: press_button('Down'))
down_button.pack(expand=True)

view_file = tk.Button(app, width=15, height=5, text="Display & Save \nhistory", fg="green",
                      command=lambda: display_history())
view_file.pack(side='left')

upload_file = tk.Button(app, width=15, height=5, text="Upload file to Drive", fg="green",
                        command=lambda: upload2drive())
upload_file.pack(side='right')

app.mainloop()
