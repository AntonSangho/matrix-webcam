import cv2

# 비디오를 녹화하는 부분
cap =  cv2.VideoCapture(0)
#bg_cap = cv2.VideoCapture('bg/bg.mp4')
#bg_cap = cv2.VideoCapture('bg/bg2.mp4')
bg_cap = cv2.VideoCapture('bg/bg3.mp4')
#bg_cap = cv2.VideoCapture('bg/bg4.mp4')
fg_cap = cv2.VideoCapture('fg/fg.mp4')

#cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
w = fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("동영상 너비: {}, 높이: {}".format(w,h))

fg_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
fg_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
w = fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("변환된 동영상 너비: {}, 높이: {}".format(w,h))

#cap_size = (int(fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap_size = (int(w), int(h))

fourcc = cv2.VideoWriter_fourcc('m', 'p','4','v')
out = cv2.VideoWriter('output.mp4', fourcc, fg_cap.get(cv2.CAP_PROP_FPS), cap_size)

# k-nearest neighbor 알고리즘을 사용해서 영상의 변화를 확인
sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=80, detectShadows=False)
# 카메라 열어서 캡쳐

while cap.isOpened():
    #ret, fg_img = cap.read()
    ret, fg_img = fg_cap.read()
    if not ret:
        break
    
    bg_ret, bg_img = bg_cap.read()

    # 영상이 끝나게 되면 첫번째 프레임으로 돌아가는 명령어
    if not bg_ret:
        bg_cap.set(1, 0)
        _, bg_img = bg_cap.read()
    
    bg_img = cv2.resize(bg_img, dsize=cap_size)

    mask = sub.apply(fg_img)

    # Morphological Transformations (형태변환)
    # 5*5 타원형태
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # morphology 의 이미지 바깥의 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # morphology 의 안쪽 바깥의 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 원본이미지를 굵어지게 만들어주는 효과
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.bitwise_and(bg_img, fg_img, mask=mask)
    #cv2.imshow('fg', fg_img)
    #cv2.imshow('bg', bg_img)
    #cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    out.write(result)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
bg_cap.release()
 
