# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:05:23 2020

@author: user
"""

import tkinter as tk
from tkinter import *
from pypylon import pylon
import cv2, os, sys, shutil

import numpy as np
import time, datetime
from PIL import Image, ImageTk
import pickle
import csv
import grpc, pickle, struct
from tkinter import filedialog

import communication_pb2
import communication_pb2_grpc


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

def connect_server():
    # serverHost = 'DESKTOP-UR7UQ1L'
    serverHost = socket.gethostname()
    port = 54321
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket Created')
        server_ip = socket.gethostbyname(serverHost)
        print('IP Address of ' + serverHost + ' is ' + server_ip)
        s.connect((server_ip, port))
        print('Socket Connected to ' + serverHost + ' on ip ' + server_ip)
        message = 'agxid'
        encodeMessage = message.encode()
        s.sendall(encodeMessage)
        print('Message send successfully')
        reply = s.recv(4096)
        get_message = reply.decode()
        return get_message
        s.close()
    except  socket.error:
        print('Failed!')


update_loop = True
def win_closing():
    global update_loop
    update_loop = False
    camera.StopGrabbing()
    # cam1.release()
    root.destroy()

root = tk.Tk()
root.title('ID識別_v1.0.0_update_20201105')
root.resizable(0, 0)

root.protocol('WM_DELETE_WINDOW', win_closing)

# =============================================================================
# 上半部
# =============================================================================

cameraFrame = tk.LabelFrame(root, text='即時畫面', fg='blue', font=('標楷體', 16))
cameraFrame.grid(column=0, row=0, padx=2, pady=2)
cameraFrame_up = tk.Frame(cameraFrame)
cameraFrame_up.grid(column=0, row=0)
screen1 = tk.Label(cameraFrame_up)
screen1.grid(column=0, row=0, padx=2, pady=2)

key = False	
if not key:	
    #initScreen = np.zeros([int(2048/4), int(2448/4)], dtype=int) # H*W #	
    initScreen = np.zeros([int(2048/4), int(2448/4)]) #linux無法採用dtype=int	
    initScreen = Image.fromarray(initScreen)	
    initScreen_tk = ImageTk.PhotoImage(image=initScreen)	
    screen1.imgtk = initScreen_tk	
    screen1.config(image=initScreen_tk)	


cameraFrame_down = tk.Frame(cameraFrame)
cameraFrame_down.grid(column=0, row=1)
camNum1 = tk.Label(cameraFrame_down, text='攝影機狀態:', font=('標楷體', 13, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
camStatus1 = tk.Label(cameraFrame_down, text='初始化', font=('標楷體', 13), fg='white', bg='orange')
camStatus1.grid(column=1, row=0, padx=2, pady=2)

show_time = tk.Label(cameraFrame, text='日期:0000-00-00 00:00:00', font=('標楷體', 13, 'bold'), fg='Coral')
show_time.grid(column=0, row=2, padx=2, pady=2)

# =============================================================================
# 下半部
# =============================================================================
infoFrame = tk.LabelFrame(root, text='系統資訊', fg='blue', font=('標楷體', 16))
infoFrame.grid(column=0, row=1, padx=2, pady=2)
left_frame = tk.Frame(infoFrame)
left_frame.grid(column=0, row=0)

currentID = tk.Label(left_frame, text='程控提供序號:', font=('標楷體', 40, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
L1_str = tk.StringVar()
currentID_box = tk.Entry(left_frame, textvariable=L1_str, bd=5, width=15, font=('標楷體', 40, 'bold'))
currentID_box.grid(column=1, row=0, padx=2, pady=2)

sysID = tk.Label(left_frame, text='系統辨識結果:', font=('標楷體', 40, 'bold'), fg='gray').grid(column=0, row=1)

second_frame = tk.Frame(left_frame, bg='red')
second_frame.grid(column=1, row=1)
idbox_str0 = tk.StringVar()
idbox0 = tk.Label(second_frame, textvariable=idbox_str0, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox0.grid(column=0, row=0)

idbox_str1 = tk.StringVar()
idbox1 = tk.Label(second_frame, textvariable=idbox_str1, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox1.grid(column=1, row=0)

idbox_str2 = tk.StringVar()
idbox2 = tk.Label(second_frame, textvariable=idbox_str2, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox2.grid(column=2, row=0)

idbox_str3 = tk.StringVar()
idbox3 = tk.Label(second_frame, textvariable=idbox_str3, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox3.grid(column=3, row=0)

idbox_str4 = tk.StringVar()
idbox4 = tk.Label(second_frame, textvariable=idbox_str4, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox4.grid(column=4, row=0)

idbox_str5 = tk.StringVar()
idbox5 = tk.Label(second_frame, textvariable=idbox_str5, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox5.grid(column=5, row=0)

idbox_str6 = tk.StringVar()
idbox6 = tk.Label(second_frame, textvariable=idbox_str6, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox6.grid(column=6, row=0)

idbox_str7 = tk.StringVar()
idbox7 = tk.Label(second_frame, textvariable=idbox_str7, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox7.grid(column=7, row=0)

idbox_str8 = tk.StringVar()
idbox8 = tk.Label(second_frame, textvariable=idbox_str8, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox8.grid(column=8, row=0)

idbox_str9 = tk.StringVar()
idbox9 = tk.Label(second_frame, textvariable=idbox_str9, bd=5, width=1, font=('標楷體', 40, 'bold'))
idbox9.grid(column=9, row=0)


# 現場ID比對
# =============================================================================
med_frame = tk.Frame(infoFrame)
med_frame.grid(column=1, row=0, padx=2, pady=2)
compare_result = tk.Label(med_frame, text='系統狀態:', font=('標楷體', 20, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
canvas = tk.Canvas(med_frame, width=100, height=100)
canvas.grid(column=0, row=1)
circle = canvas.create_oval(80, 80, 25, 25, fill='yellow')

# 建立一個list裝csv的答案
ans_lst = []
with open('./paper_ans.csv', 'r', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        ans_lst.append(row)

# ========填入csv(答案)======== # 目前是手動輸入
csv_row = 0
def input_ans():
    global csv_row
    csv_row += 1
    L1_str.set(ans_lst[csv_row][1])
    #print(csv_row)
# ============================= #
# 重寫一個 new_csv, 每當更新重寫一次
def call_new_csv():
    global ans_lst
    with open('./new_ans.csv', 'w', newline='') as newcsv:
        writer = csv.writer(newcsv)
        for i in ans_lst:
            writer.writerow(i)

button = Button(root, text='程控序號輸入下一張', font=('標楷體', 30, 'bold'), command=input_ans).grid(column=0, row=2)	
def openpicture():	
    global img, key
    key = True	
    filename = filedialog.askopenfilename()  # 獲取文件全路徑	
    print(filename)	
    img2 = Image.open(filename)	
    reimg = img2.resize((1000, 600))	
    img = ImageTk.PhotoImage(reimg)	
    	
    #img = ImageTk.PhotoImage(Image.open(filename))  # tkinter只能打開gif文件，這裏用PIL庫	
    #img = img.resize((1024, 1224), Image.ANTIALIAS)	
    # 打開jpg格式的文件
    screen1.config(image=img)  # 用config方法將圖片放置在標籤中
    img3 = cv2.imread(filename)	
    res = cv2.resize(img3, (512, 512), interpolation=cv2.INTER_CUBIC)	
    print(res.shape)	
    for i in range(5):	
        TorF, ID, BOX= remote_recognition.run(res)   	
        print(ID)	
        if len("".join(ID)) == 0:	
            canvas.itemconfig(circle, fill='red')	
        elif '?' in ID:	
            canvas.itemconfig(circle, fill='yellow')	
        else:	
            canvas.itemconfig(circle, fill='green')	
        if ID[0] != '?': idbox0.configure(fg='black')	
        else: idbox0.configure(fg='red')	
        if ID[1] != '?': idbox1.configure(fg='black')	
        else: idbox1.configure(fg='red')	
        if ID[2] != '?': idbox2.configure(fg='black')	
        else: idbox2.configure(fg='red')	
        if ID[3] != '?': idbox3.configure(fg='black')	
        else: idbox3.configure(fg='red')	
        if ID[4] != '?': idbox4.configure(fg='black')	
        else: idbox4.configure(fg='red')	
        if ID[5] != '?': idbox5.configure(fg='black')	
        else: idbox5.configure(fg='red')	
        if ID[6] != '?': idbox6.configure(fg='black')	
        else: idbox6.configure(fg='red')	
        if ID[7] != '?': idbox7.configure(fg='black')	
        else: idbox7.configure(fg='red')	
        if ID[8] != '?': idbox8.configure(fg='black')	
        else: idbox8.configure(fg='red')	
        if ID[9] != '?': idbox9.configure(fg='black')	
        else: idbox9.configure(fg='red')	
        idbox_str0.set(ID[0])	
        idbox_str1.set(ID[1])	
        idbox_str2.set(ID[2])	
        idbox_str3.set(ID[3])	
        idbox_str4.set(ID[4])	
        idbox_str5.set(ID[5])	
        idbox_str6.set(ID[6])	
        idbox_str7.set(ID[7])	
        idbox_str8.set(ID[8])	
        idbox_str9.set(ID[9])    	
        dumped_image = pickle.dumps(img2, 4)	
    
button1 = Button(root, text='讀取本機端檔案', font=('標楷體', 30, 'bold'), command=openpicture).grid(column=0, row=3)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#channel = grpc.insecure_channel('127.0.0.1:5001')
#stub = communication_pb2_grpc.dataStub(channel)
  
os.makedirs('./images/', exist_ok=True)
# ==================================================================== #

from grpc_client import RemoteRecognition
remote_recognition = RemoteRecognition()
while update_loop == True:
    root.update()
    show_time.configure(text=('日期:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():

            image = converter.Convert(grabResult)
            img = image.GetArray()
            res = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            print(res.shape)
            TorF, ID, BOX= remote_recognition.run(res)
            print(ID)
            dumped_image = pickle.dumps(img, 4)
            print(img.shape)  # 確定他是 [長, 寬, 3], max: 255, min: 0

            imgname = '{}.jpg'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))

            cv2.imwrite('./images/'+imgname, img)

            

            if len("".join(ID)) == 0:
                canvas.itemconfig(circle, fill='red')
            elif '?' in ID:
                canvas.itemconfig(circle, fill='yellow')
            else:
                canvas.itemconfig(circle, fill='green')

            if ID[0] != '?': idbox0.configure(fg='black')
            else: idbox0.configure(fg='red')
            if ID[1] != '?': idbox1.configure(fg='black')
            else: idbox1.configure(fg='red')
            if ID[2] != '?': idbox2.configure(fg='black')
            else: idbox2.configure(fg='red')
            if ID[3] != '?': idbox3.configure(fg='black')
            else: idbox3.configure(fg='red')
            if ID[4] != '?': idbox4.configure(fg='black')
            else: idbox4.configure(fg='red')
            if ID[5] != '?': idbox5.configure(fg='black')
            else: idbox5.configure(fg='red')
            if ID[6] != '?': idbox6.configure(fg='black')
            else: idbox6.configure(fg='red')
            if ID[7] != '?': idbox7.configure(fg='black')
            else: idbox7.configure(fg='red')
            if ID[8] != '?': idbox8.configure(fg='black')
            else: idbox8.configure(fg='red')
            if ID[9] != '?': idbox9.configure(fg='black')
            else: idbox9.configure(fg='red')
            idbox_str0.set(ID[0])
            idbox_str1.set(ID[1])
            idbox_str2.set(ID[2])
            idbox_str3.set(ID[3])
            idbox_str4.set(ID[4])
            idbox_str5.set(ID[5])
            idbox_str6.set(ID[6])
            idbox_str7.set(ID[7])
            idbox_str8.set(ID[8])
            idbox_str9.set(ID[9])

            ## 這邊要把 boxes 轉回去原本的 size

            ## 這邊要把 boxes 轉回去原本的 size
            
            img = cv2.resize(img, (int(2448/4), int(2048/4)), interpolation=cv2.INTER_CUBIC)
            out = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            tmpImg = Image.fromarray(out)
            if not key:	
                show = ImageTk.PhotoImage(image=tmpImg)	
                screen1.imgtk = show	
                screen1.config(image=show)	
                camStatus1.configure(text=('執行中'), bg='green')	
                time.sleep(0.5)  # 0.5	
            else:	
                camStatus1.configure(text=('讀取圖片'), bg='green')	
                time.sleep(10)	
                key == False

        else:	
            if not key:	
                screen1.imgtk = initScreen_tk	
                screen1.config(image=initScreen_tk)	
            camStatus1.configure(text=('執行錯誤'), bg='red')	
        grabResult.Release()
            



  
           
        
        
