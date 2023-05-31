import cv2

cap = cv2.VideoCapture(0)
bg_cap = cv2.VideoCapture('bg/bg3.mp4')
'''
Video source:
https://www.youtube.com/watch?v=WyKflRx56Tg
https://www.youtube.com/watch?v=cHhGjvf4jBI
https://www.youtube.com/watch?v=Dn56rvuPGZU
https://www.youtube.com/watch?v=hwSp3VYR_2o
https://www.motionstock.net/free-videos/matrix-digital-rain/
'''
# 비디오를 녹화하는 부분
cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), cap_size)

# [중요] k-nearest neighbors 알고리즘 사용하여 배경 영상의 변화를 알아내는 알고리즘 
# 과거 프레임과 최근프레임의 감지해가지고 history동안 얼마나 변했는지를 빼기 연산을 통해서 알려주는 알고리즘 
sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=100, detectShadows=False)

while cap.isOpened():
    ret, fg_img = cap.read()

    # 캠이 끝나면 break
    if not ret:
        break
    # bg 폴더에 있는 background 영상 
    bg_ret, bg_img = bg_cap.read()

    # 영상이 끝나게되면 첫번째 프레임으로 돌아가라는 명령어 
    if not bg_ret:
        bg_cap.set(1, 0)
        # 첫번째 프레임을 읽는 명령어 
        _, bg_img = bg_cap.read()

    # 사용하고 있는 웹캠과 같은 사이즈로 리사이즈를 해준다. 
    bg_img = cv2.resize(bg_img, dsize=cap_size)

    # 웹 캠 사이의 차이를 알아내서 mask에 저장한다. 
    mask = sub.apply(fg_img)

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    # 5x5 타원형태로 정의해주고 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # morphology open은 이미지 바깥의 노이즈를 없애준다
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # morphology closing은 이미지 안쪽의 노이즈를 없애준다
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 원본 이미지를 굵어지해 만들어주는 효과
    mask = cv2.dilate(mask, kernel, iterations=2)

    # bitwise_and연산을 해주게 되면 합치게 된다. 마스크 형태로
    result = cv2.bitwise_and(bg_img, fg_img, mask=mask)

    # cv2.imshow('fg', fg_img)
    # cv2.imshow('bg', bg_img)
    # mask가 어떻게 생겼는지 확인할 수 있는 부분 
    # 움지는 부분은 흰색이고 안움직이는 부분은 검정색 
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    out.write(result)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
bg_cap.release()
