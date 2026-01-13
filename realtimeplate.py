import cv2
import easyocr
import time

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
reader = easyocr.Reader(['en'] , gpu=False)
frame_count=0
last_text=""
while True:

    ret, frame = cap.read()
    if not ret:
        print("camera error!")
        break
    small=cv2.resize(frame,None,fx=0.6,fy=0.6)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY) 
    frame_count+=1
    text_found=[]

    if frame_count%10==0:
        result=reader.readtext(gray)

        for(bbox,text,prob)in result:
            if prob< 0.6:
                continue
            text_found.append(text)

            (top_left,top_right,bottom_left,bottom_right)=bbox
            x1,y1=map(int,top_left)
            x2,y2=map(int,bottom_right)

            cv2.rectangle(small,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(small,text,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        if text_found and text_found != last_text:
            print("Detected:",text_found)
            last_text = text_found

        cv2.imshow("Text Reader",small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



    