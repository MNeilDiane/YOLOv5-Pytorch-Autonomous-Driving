import threading
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from yolo import YOLO
import redis
import queue
from pypylon import pylon


class Main():

    def __init__(self):
        self.capture = cv2.VideoCapture()
        self.test_interval = 100
        self.win = tk.Tk()
        self.win.title('自动驾驶目标检测系统 V1.2')
        self.win.geometry('1280x720')
        self.win.resizable(False, False)
        self.winisvisible = 1
        self.x = 0
        self.y = 150
        self.text_queue = queue.Queue()


        canvas_root = tk.Canvas(self.win, width=1280, height=720)
        #im_root = self.get_image("car 1.png", 1280, 720)
        #canvas_root.create_image(640, 360, image=im_root)
        #canvas_root.pack()
        self.yolo = YOLO()
        redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
        self.r = redis.StrictRedis(connection_pool=redis_pool)
        self.r.hset("Communication", "Order", 0)
        self.order = self.r.hget("Communication", "Order")

        #INTEGER
        #integerl represent the detection info of rear left
        #integerr represent the detection info of rear right
        #integer represent the state info
        self.integerl = tk.StringVar()
        self.integerl.set('0')
        self.integerr = tk.StringVar()
        self.integerr.set('0')
        self.integer = tk.StringVar()
        self.integer.set('Received Message')


        #Buttons
        #im_button0 = self.get_image('7.png', 160, 100)
        self.Button0 = tk.Button(self.win, text='System Activate',
                            font=('Time New Roman', 12), bg='light blue',
                            fg='white', relief=RAISED, command=lambda:self.System_Activate())
        self.Button0.place(x=3, y=3, width=160, height=60)
        #im_button1 = self.get_image('8.png', 160, 100)

        #im_button3 = self.get_image('10.png', 160, 100)
        self.Button3 = tk.Button(self.win, text='FPS Test',
                            font=('Time New Roman', 12,),
                            bg='light blue', fg='white',
                            relief=RAISED, command=lambda: self.fps_test())
        self.Button3.place(x=180, y=3, width=160, height=60)
        #im_button4 = self.get_image('11.png', 160, 100)
        self.Button4 = tk.Button(self.win, text='Visible',font=('Time New Roman', 12),
                                 bg='light blue', fg='white',relief=RAISED, command=lambda:self.visible())
        self.Button4.place(x=350, y=3, width=160, height=60)
        #im_button5 = self.get_image('12.png', 160, 100)
        self.Button5 = tk.Button(self.win, text='Exit',
                                 font=('Time New Roman', 12), bg='light blue',
                                 fg='red', width=10, height=3, relief=RAISED,
                                 command=lambda: self.exit())
        self.Button5.place(x=520, y=3, width=160, height=60)

        self.Button6 = tk.Button(self.win, text='Shutdown',
                                 font=('Time New Roman', 12), bg='light blue',
                                 fg='yellow', width=10, height=3, relief=RAISED,
                                 command=lambda: self.shutdown()
                                 )
        self.Button6.place(x=430, y=100, width=100, height=50)

        #Labels
        #im_label1 =self.get_image('8.png', 330, 80)
        self.label1 = tk.Label(self.win, text='Detection Infomation:',
                             font=('Time New Roman', 12), bg='light blue', fg='white')
        self.label1.place(x=960, y=0, width=320, height=40)

        #Entrys
        self.Entry3 = tk.Entry(self.win, textvariable=self.integer, justify='center', font=('Time New Roman', 12),
                              bg='#c0c0c0', fg='#99ff66')
        self.Entry3.place(x=690, y=3, width=200, height=60)

        # Text
        self.text1 = tk.Text(self.win, bg='#c0c0c0', wrap=WORD)
        self.text1.place(x=960, y=40, width=320, height=680)

        #Checkbuttons
        self.Checkbutton1 = tk.Checkbutton(self.win, text='LEFT',
                                           font=('Time New Roman', 12), bg='light blue',
                                           fg='yellow', activeforeground='red',
                                           relief=RAISED, command=lambda: self.Video_left_Detection())
        self.Checkbutton1.place(x=190, y=100, width=100, height=50)
        # im_button2 = self.get_image('9.png', 160, 100)
        self.Checkbutton2 = tk.Checkbutton(self.win, text='Right',
                                           font=('Time New Roman', 12), bg='light blue',
                                           fg='yellow', activeforeground='red',
                                           relief=RAISED, command=lambda: self.Video_right_Detection())
        self.Checkbutton2.place(x=670, y=100, width=100, height=50)

        #Canvas
        self.canvas1 = tk.Canvas(self.win, bg='grey', width=480, height=300)
        self.canvas1.place(x=self.x, y=self.y)
        self.canvas2 = tk.Canvas(self.win, bg='grey', width=480, height=300)
        self.canvas2.place(x=self.x+480, y=self.y)
        self.win.mainloop()


    def System_Activate(self):
        print('System Activate!')
        self.integer.set("System Activate!")
        self.text_queue.put("System Activate--------------------------\n")
        self.update()
        self.fps_test()
        self.update()
        #while(self.order!=b'9'):
        #    po = Pool(4)
        while(self.order!=b'9'):
            thread1 = threading.Thread(target=self.Thread1_Infoextraction(), args=())
            thread2 = threading.Thread(target=self.Thread2_Reardetection(), args=())
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()






    '''
    visible i to control wheater the window is visible
    '''

    def visible(self):
        self.winisvisible=-1*self.winisvisible
        self.integer.set('winisvisle:'+str(self.winisvisible))
        print("visible:"+str(self.winisvisible))

    '''
    get_image is to adjust the image size
    '''
    def get_image(self, filename, width, height):
        im = Image.open(filename).resize((width, height))
        return ImageTk.PhotoImage(im)

    '''
    video_detection是进行视频的目标检测程序
    '''

    def Video_left_Detection(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_save_path = ""
        file = filedialog.askopenfilename()
        print("path:", file)
        video_path = file
        self.capture = cv2.VideoCapture(video_path)
        video_fps = 25.0
        size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        print('进行视频图像目标检测')
        ref, frame = self.capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        fps = 0.0
        self.text_queue.put("Rear Left Detection!")
        self.update()
        while (ref):

            t1 = time.time()
            # 读取某一帧
            ref, frame = self.capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            label = self.yolo.get_label(frame)
            self.r.hset("REAR", "LEFT INFO", str(label))
            self.integerl.set(label)
            self.text_queue.put(str(label) + '\n')
            self.update()
            print(label)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            self.order = self.r.hget("Communication", "Order")
            if self.winisvisible == 1:
                # Canvos
                frame = np.array(self.yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    2)
                pilImage = Image.fromarray(frame)
                pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                tkimage = ImageTk.PhotoImage(pilImage)
                self.canvas1.create_image(0, 0, anchor='nw', image=tkimage)
            if video_save_path != "":
                out.write(frame)
        print("Video Detection Done!")
        self.capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        #cv2.destroyAllWindows()

    def Video_right_Detection(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_save_path = ""
        file = filedialog.askopenfilename()
        print("path:", file)
        video_path = file
        self.capture = cv2.VideoCapture(video_path)
        video_fps = 25.0
        size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        print('进行视频图像目标检测')
        ref, frame = self.capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        fps = 0.0
        self.text_queue.put("Rear Right Detection!")
        self.update()
        while (ref):

            t1 = time.time()
            # 读取某一帧
            ref, frame = self.capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            label = self.yolo.get_label(frame)
            self.r.hset("REAR", "RIGHT INFO", str(label))
            self.integerl.set(label)
            self.text_queue.put(str(label) + '\n')
            self.update()
            print(label)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            self.order = self.r.hget("Communication", "Order")
            if self.winisvisible == 1:
                # Canvos
                frame = np.array(self.yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    2)
                pilImage = Image.fromarray(frame)
                pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                tkimage = ImageTk.PhotoImage(pilImage)
                self.canvas2.create_image(0, 0, anchor='nw', image=tkimage)
            if video_save_path != "":
                out.write(frame)
        print("Video Detection Done!")
        self.capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        # cv2.destroyAllWindows()
    '''
    这个程序是使用Basler相机进行实时目标检测的程序
    '''


    '''
    Update()
    '''
    def update(self):
        while not self.text_queue.empty():
            self.text1.insert(1.0, self.text_queue.get())
            self.text1.delete('17.0', '18.0')

        self.win.update()

    '''
    进行fps测试，检测机器性能
    '''

    def fps_test(self):
        img = Image.open('img/street.jpg')
        tact_time = self.yolo.get_FPS(img, self.test_interval)
        self.integer.set(str(1 / tact_time) + 'FPS')
        self.text_queue.put(str(1 / tact_time) + "FPS \n")
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    def shutdown(self):
        self.capture.release()
        self.capture1.StopGrabbing()



    def exit(self):

        if self.capture.read():
            self.capture.release()
        if self.capture1.IsGrabbing():
            self.capture1.StopGrabbing()
        sys.exit()
    '''
    Thread1 order info extraction
    '''

    def Thread1_Infoextraction(self):
        # get the order info from the redis
        self.order = self.r.hget("Communication", "Order")
        self.integer.set("Order info:" + str(self.order))
        time.sleep(0.5)
        self.text_queue.put("Waiting Order! \n")
        self.text_queue.put('\n')
        self.update()
        print("Attention!Order is :"+str(self.order))
        print("Waiting the order!")

    '''
    Thread2 wait the key to activate the  rear left and rear right detection
    '''

    def Thread2_Reardetection(self):
            time(1)

    '''
    i make reservation for the line detection and traffic lighters detection 
    '''


if __name__ == '__main__':
    Main()




