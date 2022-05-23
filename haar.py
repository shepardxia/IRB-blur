import cv2

cas = cv2.CascadeClassifier('./us.xml')
im = cv2.imread('./images/10.jpg')
result = cas.detectMultiScale(im, scaleFactor=1.2)
for (x, y, w, h) in result:
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(result)
cv2.imshow('result', im)
cv2.waitKey(0)