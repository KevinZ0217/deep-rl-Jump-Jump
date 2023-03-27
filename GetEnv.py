import win32com.client
import win32gui
import numpy as np
import pyautogui
import cv2
import time
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 缩放比例
pingmu_suofang = 1

class GetEnv:
    def __init__(self):
        self.jpg_file = 'screenshot.png'
        self.jpg_file2 = 'screenshot2.png'
        self.jpg_file3 = 'screenshot3.png'
        self.promgram_name = "跳一跳"
        self.start_btn_img = cv2.imread('./pic/img.png', cv2.IMREAD_GRAYSCALE)
        self.restart_btn_img = cv2.imread('./pic/img_1.png', cv2.IMREAD_GRAYSCALE)
        # self.agent = cv2.imread('./img.png', cv2.IMREAD_GRAYSCALE)
        self.agent_platform = cv2.imread('./img_1.png', cv2.IMREAD_GRAYSCALE)
        # 跳一跳位置
        self.tiao_x = 0
        self.tiao_y = 0 

        self.tiao_left = 0
        self.tiao_top = 0
        self.tiao_right = 0
        self.tiao_bot = 0

        # 开始游戏位置
        self.start_x = 0
        self.stat_y = 0
        self.reward_track = 0

    def set_foreground(sef, hwnd):
        """
        将窗口设置为最前面
        :param hwnd: 窗口句柄 一个整数
        """
        if hwnd != win32gui.GetForegroundWindow():
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys('%')
            win32gui.SetForegroundWindow(hwnd)

    def _find_start_btn(self, screen_shot_im, find_shot_im):
        """
        找到开始游戏位置的图标
        """
        screen_shot_im = cv2.imread(screen_shot_im, cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(screen_shot_im,
                                   find_shot_im,
                                   cv2.TM_CCOEFF_NORMED)
        if result.max() > 0.8:
            y,x = np.unravel_index(result.argmax(),result.shape)
            y += find_shot_im.shape[0] // 2
            x += find_shot_im.shape[1] // 2
            return x, y
        else:
            return -1, -1

    def find_reward(self, screen_shot, num_tbd):

        screen_shot = cv2.imread(screen_shot, cv2.IMREAD_GRAYSCALE)
        number_template = cv2.imread(f'./pic/template_{num_tbd}.png', cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(screen_shot, number_template, cv2.TM_CCOEFF_NORMED)
        number = -1
        if result.max() > 0.95:
           number = num_tbd
        return number

        # text = pytesseract.image_to_string(screen_shot, config="--psm 6")
        # print(f"reward_test: {text}")
        # return int(text)

    def screen_pic(self, promgram_name, jpg_file, jpg_file2):
        """
        截屏窗口图
        """
        hWnd = win32gui.FindWindow(None, promgram_name) #窗口的类名可以用Visual Studio的SPY++工具获取
        # 设置窗口在最前面
        self.set_foreground(hWnd)
        time.sleep(1)
        left, top, right, bot = win32gui.GetWindowRect(hWnd)
        self.left = left
        self.top = top+60
        self.right = right
        self.bot = bot
        self.tiao_x = (right + left) // 2
        self.tiao_y = (bot + top) // 2

        # 截屏
        img = pyautogui.screenshot(region=None) # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_file, img)

        # 识别“点击开始游戏”位置
        self.start_x, self.stat_y = self._find_start_btn(jpg_file, self.start_btn_img)
        pyautogui.click(x=self.start_x, y=self.stat_y, duration=0.25)

        time.sleep(1)

        # 重新截屏
        img = pyautogui.screenshot(region=None) # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_file, img)

        # 剪裁
        rangle = (int(left) * pingmu_suofang, int(top) * pingmu_suofang, int(right) * pingmu_suofang, int(bot) * pingmu_suofang)

        img = Image.open(jpg_file)
        jpg = img.convert('RGB')
        jpg = img.crop(rangle)
        jpg.save(jpg_file2)
        # jpg.show()

    def get_gray_image(self, jpg_file2, is_show=False):
        """
        转成灰度图，并进行归一化, numpy(200, 120)
        """
        im = cv2.imread(jpg_file2, cv2.IMREAD_GRAYSCALE)
        
        
        if is_show:
            cv2.imshow("name:img_gray", im)
            cv2.waitKey(0)   
            cv2.destroyAllWindows()

        im = np.expand_dims(cv2.resize(im, (120, 200)) / 255.0, -1)
        im = np.float32(np.expand_dims(im, 0))
        return im

    def defeate_return():
        """
        失败返回
        """


    def reset(self, is_show=False):
        """
        初始化环境
        """

        # 截屏
        self.screen_pic(self.promgram_name, self.jpg_file, self.jpg_file2)
        # 转灰度图
        image_gray = self.get_gray_image(self.jpg_file2, is_show)

        return image_gray

    def dist(self, action):
        """
        计算按压时间:
        return: 最低按压0.3s，最高按压1.1s
        """
        d = action * 400 + 700
        if d < 300:
            d = 300
        elif d > 1000:
            d = 1000
            
        return d / 1000

    def touch_in_step(self, action):
        """
        按压时间: ms
        """

        time.sleep(0.5)
        
        # 计算按压时间
        dist = self.dist(action.numpy()[0, 0])#dist = self.dist(action)
        print(f"pressing time: {dist}")
        #print("dist:", dist)
        pyautogui.moveTo(x=self.tiao_x, y=self.tiao_y, duration=0.25)
        pyautogui.dragTo(x=self.tiao_x, y=self.tiao_y, duration=dist)
        
        time.sleep(dist+3.5)

        # 截屏
        ## 重新截屏
        img = pyautogui.screenshot(region=None) # x,y,w,h
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.jpg_file, img)

        ## 剪裁
        rangle = (int(self.left) * pingmu_suofang, int(self.top) * pingmu_suofang, int(self.right) * pingmu_suofang, int(self.bot) * pingmu_suofang)
        reward_rangle = (int(self.left) * pingmu_suofang, int(self.top) * pingmu_suofang, int(self.right) * pingmu_suofang, int(self.bot/2-20) * pingmu_suofang)
        reward_img = Image.open(self.jpg_file)
        reward_jpg = reward_img.convert('RGB')
        reward_jpg = reward_jpg.crop(reward_rangle)
        reward_jpg.save(self.jpg_file3)
        img = Image.open(self.jpg_file)
        jpg = img.convert('RGB')
        jpg = jpg.crop(rangle) # to be fixed
        jpg.save(self.jpg_file2)

        # 查找是否失败
        time.sleep(2)
        restart_x, restart_y = self._find_start_btn(self.jpg_file, self.restart_btn_img)
        print('cursor location:', restart_x)

        # The game has not ended yet
        if restart_x == -1:
            reward = 0
            number = -1
            # num_list = []
            # for i in range(10):
            #     number = self.find_reward(self.jpg_file3, i)
            #     print(f"number:{number}")
            #     if number != -1:
            #         num_list.append(number)
            #         number = -1
            # print(f'length numlist: {len(num_list)}')
            # if len(num_list) == 2:
            #     curr_reward_1, curr_reward_2 = num_list[0]*10 + num_list[1], num_list[1]*10 + num_list[0]
            #     if curr_reward_1 > self.reward_track:
            #         reward = curr_reward_1
            #         self.reward_track = reward
            #     else:
            #         reward = curr_reward_2
            #         self.reward_track = reward
            # else:
            #     curr_reward_1 = num_list[0]
            #     reward = curr_reward_1
            #     self.reward_track = curr_reward_1


            state = self.get_gray_image(self.jpg_file2)

            # Convert the image to grayscale
            #image = cv2.imread(self.jpg_file2)
            #template_ = cv2.imread(self.agent_platform)
            # agent_x, agent_y = self._find_start_btn(self.jpg_file2, self.agent)
            # contour detection
            # img_blur = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
            # canny_img = cv2.Canny(img_blur, 1, 10)  # 边缘检测
            #screen_shot_im = cv2.imread(self.jpg_file2, cv2.IMREAD_GRAYSCALE)
            #ind_shot_im = cv2.imread(template_, cv2.IMREAD_GRAYSCALE)
            #result = cv2.matchTemplate(screen_shot_im,
            #                           self.agent_platform,
            #                           cv2.TM_CCOEFF_NORMED)
           # dist = result.max()
            #print(dist)
            reward = 1
            # #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # # Apply a threshold to create a binary image
            # #_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # # Find contours in the binary image
            # contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # center_locations = []
            # # Iterate through the contours and calculate the center of each contour
            # for contour in contours:
            #     # Calculate the moments of the contour
            #     moments = cv2.moments(contour)
            #
            #     # Calculate the center of the contour
            #     if moments['m00'] != 0:
            #         center_x = int(moments['m10'] / moments['m00'])
            #         center_y = int(moments['m01'] / moments['m00'])
            #     else:
            #         center_x, center_y = 0, 0
            #     center_locations.append((center_x, center_y))
            #     # Draw a circle at the center of the contour (optional)
            #     cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
            # # Display the original image with the center of the contour marked (optional)
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(center_locations)

            done = False
            return state, reward, done
        else:
            # 失败返回
            pyautogui.click(x=restart_x, y=restart_y, duration=0.4)
            time.sleep(2)
            state = None
            reward = -1
            done = True
            return state, reward, done

if __name__ == '__main__':
    get_env = GetEnv()
    res = get_env.reset(is_show=False)
    start_time = time.time()
    for i in range(10):
        print("%d Use time %f" % (i, time.time() - start_time))
        next_state, reward, done = get_env.touch_in_step(10)

        if done:
            res = get_env.reset(is_show=False)
        else:
            break


