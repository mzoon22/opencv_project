import cv2

# 얼굴 검출을 위한 Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier('./haar_face.xml')
# 이미지 또는 비디오에서 얼굴 검출
img = cv2.imread('./image/7.jpg') #jpg,jpeg,png 중에 파일인지 잘 확인 후 기입
if img is None:
    print("이미지를 제대로 읽을 수 없습니다.")

gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.3, minNeighbors=5)

# 고정된 크기로 얼굴 영역 조절
fixed_size = (100, 100)  # 원하는 크기로 조절
for (x, y, w, h) in faces:
   gray_mouth = gray_face[y:y+h, x:x+w]
   resized_face = cv2.resize(gray_mouth, fixed_size)
   
   mouth_cascade = cv2.CascadeClassifier('./haar_mouth.xml')
   mouth = mouth_cascade.detectMultiScale(gray_mouth, scaleFactor=1.3, minNeighbors=5)

   if len(mouth) > 0:
     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 빨간색 사각형: 마스크 미착용
   else:
     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록색 사각형: 마스크 착용

# 결과 표시
cv2.imshow('Mask Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
