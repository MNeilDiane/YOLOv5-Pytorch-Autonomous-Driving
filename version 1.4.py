import threading
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from yolo import YOLO
from pyueye import ueye
import redis
import queue


class Main():

    def __init__(self):
        self.capture = cv2.VideoCapture()
        self.hCam0 = ueye.HIDS(1001)
        self.hCam1 = ueye.HIDS(1002)
        self.test_interval = 100
        self.win = tk.Tk()
        self.win.title('自动驾驶目标检测系统 V1.3')
        self.win.geometry('1280x720')
        self.win.resizable(False, False)
        self.winisvisible = 1
        self.stop = False
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
        while(self.order != b'9'):
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
        self.stop = False
        if self.stop == False:
            self.hCam0 = ueye.HIDS(1)
            # first available camera  1-254 The camera with its specified ID
            mem_ptr = ueye.c_mem_p()  # pcImageMemory
            mem_id = ueye.int()  # MEM_ID
            # Starts the driver and establishes the connection to the camera
            ret = ueye.is_InitCamera(self.hCam0, None)
            if ret != ueye.IS_SUCCESS:
                print('init camera failed')
            else:
                print('init camera success')
            print("Rear Left Detection!")
            self.text_queue.put("Rear Left Detection!\n")
            self.win.update()
            rangMin = ueye.double()
            rangMax = ueye.double()
            # increment
            rangInc = ueye.double()
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, rangMin, ueye.sizeof(rangMin))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, rangMax, ueye.sizeof(rangMax))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, rangInc, ueye.sizeof(rangInc))
            print('rangMin:' + str(rangMin))
            print('rangMax:' + str(rangMax))
            print('rangInc:' + str(rangInc))
            # set exposure time in the range
            exposTime = ueye.double(rangMin + 10)  # any value you want in the range
            print('ExposTime:' + str(exposTime))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposTime, ueye.sizeof(exposTime))
            # set diaplay mode
            ret = ueye.is_SetDisplayMode(self.hCam0, ueye.IS_SET_DM_DIB)
            # set color mode
            # ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
            nbpp = 24  # bits of per pixel. this value is associated with the color mode

            # get image size
            rect_aoi = ueye.IS_RECT()
            # Can be used to set the size and position of an "area of interest"(AOI) within an image
            ueye.is_AOI(self.hCam0, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            # allocate memory
            ret = ueye.is_AllocImageMem(self.hCam0,
                                        rect_aoi.s32Width.value,
                                        rect_aoi.s32Height.value,
                                        nbpp,
                                        mem_ptr,
                                        mem_id,
                                        )
            # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
            if ret != ueye.IS_SUCCESS:
                print('allocate image memory failed')
            else:
                print('allocate memory')
            # the allocated memory must be actived by set iamge
            ret = ueye.is_SetImageMem(self.hCam0, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                print('set image memory failed')
            else:
                print('set image memory')
                flag = True
                count = 0
                fps = 0.0

                while flag:
                    # is_FreezeVideo excute once, capture one image
                    ret = ueye.is_FreezeVideo(self.hCam0, ueye.IS_WAIT)
                    # ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)4
                    if ret != ueye.IS_SUCCESS:
                        print('capture failed')
                    else:
                        fps = 0.0
                        t1 = time.time()
                        count += 1
                        print('capture %d images' % (count))
                        # print('capture %d images' %(count))
                        # format memory data to OpenCV Mat
                        # extract the data of our image memory
                        # ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                        array = ueye.get_data(mem_ptr, rect_aoi.s32Width.value, rect_aoi.s32Height.value, nbpp,
                                              rect_aoi.s32Width.value * int((nbpp + 7) / 8), True)
                        frame = np.reshape(array, (rect_aoi.s32Height.value, rect_aoi.s32Width.value, 3))
                        frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
                        # 格式转变，BGRtoRGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 转变成Image
                        frame = Image.fromarray(np.uint8(frame))
                        label = self.yolo.get_label(frame)
                        fps = (fps + (1. / (time.time() - t1))) / 2
                        print("Fps:" + str(fps * 2))
                        self.r.hset("REAR", "LEFT INFO", str(label))
                        self.text_queue.put(str(label) + '\n')
                        self.text_queue.put('\n')
                        self.update()
                        self.order = self.r.hget("Communication", "Order")
                        if self.stop != False:
                            flag = False
                            ueye.is_FreeImageMem(self.hCam0, mem_ptr, mem_id)
                            mem_ptr = None
                            ueye.is_ExitCamera(self.hCam0)
                            break
                        if self.winisvisible == 1:
                            # 进行检测
                            frame = np.array(self.yolo.detect_image(frame))
                            # RGBtoBGR满足opencv显示格式
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            # print("fps= %.2f" % (fps))
                            frame = cv2.putText(frame, "fps= %.2f" % (fps * 2), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0), 2)
                            pilImage = Image.fromarray(frame)
                            pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                            tkimage = ImageTk.PhotoImage(pilImage)
                            self.canvas1.create_image(0, 0, anchor='nw', image=tkimage)

    def Video_right_Detection(self):
        self.stop = False
        if self.stop == False:
            self.hCam0 = ueye.HIDS(1)
            # first available camera  1-254 The camera with its specified ID
            mem_ptr = ueye.c_mem_p()  # pcImageMemory
            mem_id = ueye.int()  # MEM_ID
            # Starts the driver and establishes the connection to the camera
            ret = ueye.is_InitCamera(self.hCam0, None)
            if ret != ueye.IS_SUCCESS:
                print('init camera failed')
            else:
                print('init camera success')
            print("Rear Left Detection!")
            self.text_queue.put("Rear Left Detection!\n")
            self.win.update()
            rangMin = ueye.double()
            rangMax = ueye.double()
            # increment
            rangInc = ueye.double()
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, rangMin, ueye.sizeof(rangMin))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, rangMax, ueye.sizeof(rangMax))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, rangInc, ueye.sizeof(rangInc))
            print('rangMin:' + str(rangMin))
            print('rangMax:' + str(rangMax))
            print('rangInc:' + str(rangInc))
            # set exposure time in the range
            exposTime = ueye.double(rangMin + 10)  # any value you want in the range
            print('ExposTime:' + str(exposTime))
            ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposTime, ueye.sizeof(exposTime))
            # set diaplay mode
            ret = ueye.is_SetDisplayMode(self.hCam0, ueye.IS_SET_DM_DIB)
            # set color mode
            # ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
            nbpp = 24  # bits of per pixel. this value is associated with the color mode

            # get image size
            rect_aoi = ueye.IS_RECT()
            # Can be used to set the size and position of an "area of interest"(AOI) within an image
            ueye.is_AOI(self.hCam0, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            # allocate memory
            ret = ueye.is_AllocImageMem(self.hCam0,
                                        rect_aoi.s32Width.value,
                                        rect_aoi.s32Height.value,
                                        nbpp,
                                        mem_ptr,
                                        mem_id,
                                        )
            # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
            if ret != ueye.IS_SUCCESS:
                print('allocate image memory failed')
            else:
                print('allocate memory')
            # the allocated memory must be actived by set iamge
            ret = ueye.is_SetImageMem(self.hCam0, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                print('set image memory failed')
            else:
                print('set image memory')
                flag = True
                count = 0
                fps = 0.0

                while flag:
                    # is_FreezeVideo excute once, capture one image
                    ret = ueye.is_FreezeVideo(self.hCam0, ueye.IS_WAIT)
                    # ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)4
                    if ret != ueye.IS_SUCCESS:
                        print('capture failed')
                    else:
                        fps = 0.0
                        t1 = time.time()
                        count += 1
                        print('capture %d images' % (count))
                        # print('capture %d images' %(count))
                        # format memory data to OpenCV Mat
                        # extract the data of our image memory
                        # ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                        array = ueye.get_data(mem_ptr, rect_aoi.s32Width.value, rect_aoi.s32Height.value, nbpp,
                                              rect_aoi.s32Width.value * int((nbpp + 7) / 8), True)
                        frame = np.reshape(array, (rect_aoi.s32Height.value, rect_aoi.s32Width.value, 3))
                        frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
                        # 格式转变，BGRtoRGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 转变成Image
                        frame = Image.fromarray(np.uint8(frame))
                        label = self.yolo.get_label(frame)
                        fps = (fps + (1. / (time.time() - t1))) / 2
                        print("Fps:" + str(fps * 2))
                        self.r.hset("REAR", "LEFT INFO", str(label))
                        self.text_queue.put(str(label) + '\n')
                        self.text_queue.put('\n')
                        self.update()
                        self.order = self.r.hget("Communication", "Order")
                        if self.stop != False:
                            flag = False
                            ueye.is_FreeImageMem(self.hCam0, mem_ptr, mem_id)
                            mem_ptr = None
                            ueye.is_ExitCamera(self.hCam0)
                            break
                        if self.winisvisible == 1:
                            # 进行检测
                            frame = np.array(self.yolo.detect_image(frame))
                            # RGBtoBGR满足opencv显示格式
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            # print("fps= %.2f" % (fps))
                            frame = cv2.putText(frame, "fps= %.2f" % (fps * 2), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0), 2)
                            pilImage = Image.fromarray(frame)
                            pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                            tkimage = ImageTk.PhotoImage(pilImage)
                            self.canvas2.create_image(0, 0, anchor='nw', image=tkimage)

    '''
    这个程序是使用IDS相机进行实时目标检测的程序
    '''

    def camera_detection(self):
        print('进行摄像头实时目标检测')
        hCam = ueye.HIDS(0)  # first available camera  1-254 The camera with its specified ID
        mem_ptr = ueye.c_mem_p()  # pcImageMemory
        mem_id = ueye.int()  # MEM_ID
        # Starts the driver and establishes the connection to the camera
        ret = ueye.is_InitCamera(hCam, None)
        if ret != ueye.IS_SUCCESS:
            print('init camera failed')
        else:
            print('init camera success')

        rangMin = ueye.double()
        rangMax = ueye.double()
        # increment
        rangInc = ueye.double()
        ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, rangMin, ueye.sizeof(rangMin))
        ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, rangMax, ueye.sizeof(rangMax))
        ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, rangInc, ueye.sizeof(rangInc))
        print('rangMin:' + str(rangMin))
        print('rangMax:' + str(rangMax))
        print('rangInc:' + str(rangInc))
        # set exposure time in the range
        exposTime = ueye.double(rangMin + 10)  # any value you want in the range
        print('ExposTime:' + str(exposTime))
        ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposTime, ueye.sizeof(exposTime))

        # set diaplay mode
        ret = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
        # set color mode
        # ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
        nbpp = 24  # bits of per pixel. this value is associated with the color mode

        # get image size
        rect_aoi = ueye.IS_RECT()
        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        # allocate memory
        ret = ueye.is_AllocImageMem(hCam,
                                    rect_aoi.s32Width.value,
                                    rect_aoi.s32Height.value,
                                    nbpp,
                                    mem_ptr,
                                    mem_id,
                                    )
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        if ret != ueye.IS_SUCCESS:
            print('allocate image memory failed')
        else:
            print('allocate memory')

        # the allocated memory must be actived by set iamge
        ret = ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
        if ret != ueye.IS_SUCCESS:
            print('set image memory failed')
        else:
            print('set image memory')
            flag = True
            count = 0
            fps = 0.0

            while flag:
                # is_FreezeVideo excute once, capture one image
                ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
                # ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)4
                if ret != ueye.IS_SUCCESS:
                    print('capture failed')
                else:
                    start_time = time.time()
                    count += 1
                    print('capture %d images' % (count))
                    # print('capture %d images' %(count))
                    # format memory data to OpenCV Mat
                    # extract the data of our image memory
                    # ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                    array = ueye.get_data(mem_ptr, rect_aoi.s32Width.value, rect_aoi.s32Height.value, nbpp,
                                          rect_aoi.s32Width.value * int((nbpp + 7) / 8), True)
                    frame = np.reshape(array, (rect_aoi.s32Height.value, rect_aoi.s32Width.value, 3))
                    frame = cv2.resize(frame, (480, 300), interpolation=cv2.INTER_CUBIC)
                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转变成Image
                    frame = Image.fromarray(np.uint8(frame))
                    if self.winisvisible == 1:
                        # 进行检测
                        frame = np.array(self.yolo.detect_image(frame))
                        # RGBtoBGR满足opencv显示格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        print('FPS: ', 1.0 / (time.time() - start_time), "fps")
                        fps = (fps + (1. / (time.time() - start_time))) / 2
                        # print("fps= %.2f" % (fps))
                        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                        pilImage = Image.fromarray(frame)
                        pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                        tkimage = ImageTk.PhotoImage(pilImage)
                        self.canvas1.create_image(0, 0, anchor='nw', image=tkimage)

            # close camera
            if mem_ptr != None:
                ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
                mem_ptr = None
            ueye.is_ExitCamera(hCam)
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
        self.stop = True



    def exit(self):

        if self.capture.read():
            self.capture.release()
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
        while(True):
            if self.order == b'1':
                #first available camera  1-254 The camera with its specified ID
                mem_ptr = ueye.c_mem_p()  # pcImageMemory
                mem_id = ueye.int()  # MEM_ID
                # Starts the driver and establishes the connection to the camera
                ret = ueye.is_InitCamera(self.hCam0, None)
                if ret != ueye.IS_SUCCESS:
                    print('init camera failed')
                else:
                    print('init camera success')
                print("Rear Left Detection!")
                self.text_queue.put("Rear Left Detection!\n")
                self.win.update()
                rangMin = ueye.double()
                rangMax = ueye.double()
                # increment
                rangInc = ueye.double()
                ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, rangMin, ueye.sizeof(rangMin))
                ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, rangMax, ueye.sizeof(rangMax))
                ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, rangInc, ueye.sizeof(rangInc))
                print('rangMin:' + str(rangMin))
                print('rangMax:' + str(rangMax))
                print('rangInc:' + str(rangInc))
                # set exposure time in the range
                exposTime = ueye.double(rangMin + 10)  # any value you want in the range
                print('ExposTime:' + str(exposTime))
                ueye.is_Exposure(self.hCam0, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposTime, ueye.sizeof(exposTime))
                # set diaplay mode
                ret = ueye.is_SetDisplayMode(self.hCam0, ueye.IS_SET_DM_DIB)
                # set color mode
                # ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
                nbpp = 24  # bits of per pixel. this value is associated with the color mode

                # get image size
                rect_aoi = ueye.IS_RECT()
                # Can be used to set the size and position of an "area of interest"(AOI) within an image
                ueye.is_AOI(self.hCam0, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
                # allocate memory
                ret = ueye.is_AllocImageMem(self.hCam0,
                                            rect_aoi.s32Width.value,
                                            rect_aoi.s32Height.value,
                                            nbpp,
                                            mem_ptr,
                                            mem_id,
                                            )
                # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
                if ret != ueye.IS_SUCCESS:
                    print('allocate image memory failed')
                else:
                    print('allocate memory')
                # the allocated memory must be actived by set iamge
                ret = ueye.is_SetImageMem(self.hCam0, mem_ptr, mem_id)
                if ret != ueye.IS_SUCCESS:
                    print('set image memory failed')
                    if self.order != b'1':
                        break

                else:
                    print('set image memory')
                    flag = True
                    count = 0
                    fps = 0.0


                    while flag:
                        # is_FreezeVideo excute once, capture one image
                        ret = ueye.is_FreezeVideo(self.hCam0, ueye.IS_WAIT)
                        # ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)4
                        if ret != ueye.IS_SUCCESS:
                            print('capture failed')
                        else:
                            fps = 0.0
                            t1 = time.time()
                            count += 1
                            print('capture %d images' % (count))
                            # print('capture %d images' %(count))
                            # format memory data to OpenCV Mat
                            # extract the data of our image memory
                            # ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                            array = ueye.get_data(mem_ptr, rect_aoi.s32Width.value, rect_aoi.s32Height.value, nbpp,
                                                  rect_aoi.s32Width.value * int((nbpp + 7) / 8), True)
                            frame = np.reshape(array, (rect_aoi.s32Height.value, rect_aoi.s32Width.value, 3))
                            frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
                            # 格式转变，BGRtoRGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # 转变成Image
                            frame = Image.fromarray(np.uint8(frame))
                            label = self.yolo.get_label(frame)
                            fps = (fps + (1. / (time.time() - t1))) / 2
                            print("Fps:"+str(fps*2))
                            self.r.hset("REAR", "LEFT INFO", str(label))
                            self.text_queue.put(str(label) + '\n')
                            self.text_queue.put('\n')
                            self.update()
                            self.order = self.r.hget("Communication", "Order")
                            if self.order != b'1':
                                flag = False
                                ueye.is_FreeImageMem(self.hCam0, mem_ptr, mem_id)
                                mem_ptr = None
                                ueye.is_ExitCamera(self.hCam0)
                                break
                            if self.winisvisible == 1:
                                # 进行检测
                                frame = np.array(self.yolo.detect_image(frame))
                                # RGBtoBGR满足opencv显示格式
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                # print("fps= %.2f" % (fps))
                                frame = cv2.putText(frame, "fps= %.2f" % (fps*2), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)
                                pilImage = Image.fromarray(frame)
                                pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                                tkimage = ImageTk.PhotoImage(pilImage)
                                self.canvas1.create_image(0, 0, anchor='nw', image=tkimage)
            if self.order == b'-1':
                mem_ptr = ueye.c_mem_p()  # pcImageMemory
                mem_id = ueye.int()  # MEM_ID
                # Starts the driver and establishes the connection to the camera
                ret = ueye.is_InitCamera(self.hCam1, None)
                if ret != ueye.IS_SUCCESS:
                    print('init camera failed')
                else:
                    print('init camera success')
                print("Rear Right Detection!")
                self.text_queue.put("Rear Right Detection!\n")
                self.win.update()
                rangMin = ueye.double()
                rangMax = ueye.double()
                # increment
                rangInc = ueye.double()
                ueye.is_Exposure(self.hCam1, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, rangMin, ueye.sizeof(rangMin))
                ueye.is_Exposure(self.hCam1, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, rangMax, ueye.sizeof(rangMax))
                ueye.is_Exposure(self.hCam1, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, rangInc, ueye.sizeof(rangInc))
                print('rangMin:' + str(rangMin))
                print('rangMax:' + str(rangMax))
                print('rangInc:' + str(rangInc))
                # set exposure time in the range
                exposTime = ueye.double(rangMin + 10)  # any value you want in the range
                print('ExposTime:' + str(exposTime))
                ueye.is_Exposure(self.hCam1, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposTime, ueye.sizeof(exposTime))
                # set diaplay mode
                ret = ueye.is_SetDisplayMode(self.hCam1, ueye.IS_SET_DM_DIB)
                # set color mode
                # ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
                nbpp = 24  # bits of per pixel. this value is associated with the color mode

                # get image size
                rect_aoi = ueye.IS_RECT()
                # Can be used to set the size and position of an "area of interest"(AOI) within an image
                ueye.is_AOI(self.hCam1, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
                # allocate memory
                ret = ueye.is_AllocImageMem(self.hCam1,
                                            rect_aoi.s32Width.value,
                                            rect_aoi.s32Height.value,
                                            nbpp,
                                            mem_ptr,
                                            mem_id,
                                            )
                # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
                if ret != ueye.IS_SUCCESS:
                    print('allocate image memory failed')
                else:
                    print('allocate memory')
                # the allocated memory must be actived by set iamge
                ret = ueye.is_SetImageMem(self.hCam1, mem_ptr, mem_id)
                if ret != ueye.IS_SUCCESS:
                    print('set image memory failed')
                    if self.order != -1:
                        break
                else:
                    print('set image memory')
                    flag = True
                    count = 0
                    fps = 0.0

                    while flag:
                        # is_FreezeVideo excute once, capture one image
                        ret = ueye.is_FreezeVideo(self.hCam1, ueye.IS_WAIT)
                        # ret = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)4
                        if ret != ueye.IS_SUCCESS:
                            print('capture failed')
                        else:
                            start_time = time.time()
                            count += 1
                            print('capture %d images' % (count))
                            # print('capture %d images' %(count))
                            # format memory data to OpenCV Mat
                            # extract the data of our image memory
                            # ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
                            array = ueye.get_data(mem_ptr, rect_aoi.s32Width.value, rect_aoi.s32Height.value, nbpp,
                                                  rect_aoi.s32Width.value * int((nbpp + 7) / 8), True)
                            frame = np.reshape(array, (rect_aoi.s32Height.value, rect_aoi.s32Width.value, 3))
                            frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
                            # 格式转变，BGRtoRGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # 转变成Image
                            frame = Image.fromarray(np.uint8(frame))
                            label = self.yolo.get_label(frame)
                            self.r.hset("REAR", "RIGHT INFO", str(label))
                            self.text_queue.put(str(label) + '\n')
                            self.text_queue.put('\n')
                            self.update()
                            self.order = self.r.hget("Communication", "Order")
                            if self.order != b'-1':
                                flag = False
                                ueye.is_FreeImageMem(self.hCam1, mem_ptr, mem_id)
                                mem_ptr = None
                                ueye.is_ExitCamera(self.hCam1)
                                break
                            if self.winisvisible == 1:
                                # 进行检测
                                frame = np.array(self.yolo.detect_image(frame))
                                # RGBtoBGR满足opencv显示格式
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                                print('FPS: ', 1.0 / (time.time() - start_time), "fps")
                                fps = (fps + (1. / (time.time() - start_time))) / 2
                                # print("fps= %.2f" % (fps))
                                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)
                                pilImage = Image.fromarray(frame)
                                pilImage = pilImage.resize((480, 300), Image.ANTIALIAS)
                                tkimage = ImageTk.PhotoImage(pilImage)
                                self.canvas2.create_image(0, 0, anchor='nw', image=tkimage)
            else:
                break



    '''
    i make reservation for the line detection and traffic lighters detection 
    '''


if __name__ == '__main__':
    Main()




