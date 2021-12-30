import cv2
import numpy as np

# 모자이크 죽소 Ratio설정
# 최소값 0.2
print("Before We Start")
ratio = float(input("mosa ratio : "))

# 학습된 YOLO 읽어오는 부분
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# 클래스의 갯수만큼 랜덤 RGB 배열을 생성
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

print("[INFO] accessing video stream...")
capture = cv2.VideoCapture("C:/Users/ㅎㅈ/PycharmProjects/pythonProject/venv/21_person.mp4")
frame_per_second = (capture.get(cv2.CAP_PROP_POS_FRAMES))

# process함수를 사용위해 필요
frame_per_second += 1
ret, frame = capture.read()


# 프레임 조절
# 영상을 실시간으로 볼 때, 너무 느리면 사용
def process(Time):
    count = 1
    while True:
        ret, frame = capture.read()
        if count % (Time * frame_per_second) == 0:
            break
        count += 1


# 마우스를 영역을 지정하여 모자이크 처리
# Q버튼 입력 또는 모자이크 resize오류 시 동작
def roi():
    x, y, w, h = cv2.selectROI(frame, False)
    if w and h:
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        frame[y:y + h, x:x + w] = roi


# Video를 저장하기 위해 Video Codec, fps, Writer 설정
print("[INFO] Saving Video...")
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mov', fourcc, fps, (720, 480))

if not capture.isOpened():
    print("Could not open video")
    exit()

while True:
    # Video를 읽어오는 부분
    if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("C:/Users/ㅎㅈ/PycharmProjects/pythonProject/venv/21_person.mp4")

    ret, frame = capture.read()

    frame = cv2.resize(frame, (720, 480))

    # process(3)
    if not ret:
        print('error')
        break
    # Frame의 높이, 넓이, 채널을 받아옴
    height, width, channels = frame.shape

    if not ret:
        print("Could not read video")
        exit()

    # Detecting objects / 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # Showing informations on the screen
    # 각 각의 정보를 저장할 빈 리스트 선언
    class_ids = []
    confidences = []
    boxes = []
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # confidence를 낮춰서 민감도 조절
            if confidence > 0.2:
                # Object detected
                # Detect된 객체의 넓이, 높이, 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NonMaximum Supperssion
    # 겹쳐있는 박스 중 Confidence가 가장 높은 Rectangle을 선택
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 박스 중 선택된 박스의 인텍스 출력
    # print([INFO] indexes)
    font = cv2.FONT_HERSHEY_PLAIN

    # 좌표값을 저장하기 위한 빈 리스트 선언
    # [x,y,w,h] 1 By 4로 저장할 것이기 때문에 어떻게 저장할지 form을 미리 선언해주어야 함
    far_persons_list = np.empty((0, 4), int)
    near_persons_list = np.empty((0, 4), int)
    move_list = np.empty((0, 4), int)
    rights_list = np.empty((0, 4), int)
    lefts_list = np.empty((0, 4), int)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # print('[INFO] Rectangle Coordinate Saving')

            # Detecting된 Object의 Label이 Person일 때
            if 'person' in label:
                color = (0, 0, 255)
                # 멀리 있는 사람과 가까이 있는 사람에 대해서 Rectangle
                # 크기를 다르게 하기 위해 영역을 나눔
                if 50 < y < 80:
                    # 사람의 Rectangle에서 사람의 머리 부분만 저장
                    # far_persons = cv2.rectangle(frame, (x, y), (x + w, (y + 25)), color, 1)
                    far_persons_list = np.append(far_persons_list, np.array([[x, y, x + w, y + 25]]), axis=0)
                    # print('far_persons_list : %s' %far_persons_list)

                elif y > 60:
                    # person의 Rectangle에서 사람의 머리 부분만 저장
                    # near_persons = cv2.rectangle(frame, (x, y), (x + w, (y + 60)), color, 1)
                    near_persons_list = np.append(near_persons_list, np.array([[x, y, x + w, y + 60]]), axis=0)
                    # print('near_persons_list : %s' %near_persons_list)

            # Detecting된 Object의 Label이 Car일 때
            elif 'car' in label:
                if y > 75:
                    if 150 < x < 475:
                        # 움직이는 차만 Detecting후 리스트에 저장
                        color = (0, 0, 120)
                        # move = cv2.rectangle(frame, (x, y+30), (x+w, y+h), color, 1)
                        move_list = np.append(move_list, np.array([[x, y + 30, x + w, y + h]]), axis=0)
                        # print('move_list : %s' %move_list)

                    elif x > 470:
                        # 오른쪽에 있는 차만 Detecting후 리스트에 저장
                        color = (255, 0, 0)
                        # rights = cv2.rectangle(frame, (x, y+35), (x+40, y+120), color, 1)
                        rights_list = np.append(rights_list, np.array([[x, y + 35, x + 40, y + 120]]), axis=0)
                        # print('rights_list : %s' %rights_list)

                    elif x < 250:
                        # 왼쪽에 있는 차만 Detecting후 리스트에 저장
                        color = (0, 255, 0)
                        # lefts = cv2.rectangle(frame, (x+90, y+20), (x+w, y+140), color, 1)
                        lefts_list = np.append(lefts_list, np.array([[x + 90, y + 20, x + w, y + 140]]), axis=0)
                        # print('lefts_list : %s' %lefts_list)

    # 왼쪽, 오른쪽 차량 Detecting된 갯수가 2개 이하일 때 Roi함수 실행
    # if len(rights_list) < 2 :
    #     roi()

    # if len(rights_list) < 2 :
    #     roi()

    # if len(rights_list) != 0:
    # for (rightsIDs, centroids) in rights.items():
    # check = False

    # for (rightsID, centroid) in rights.items():
    #     text = "ID{}".format(rightsID)
    #     cv2.putText(frame. text, )centroid[0] -10, centroid[1] -10, cv2.FONT_HERSHEY_SIMPLEX, 0,5, (0,255,0
    #         0,2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
    #     print("ID count : ", len(rights.items()))

    # Detect한 Rectangle 모자이크 처리
    try:
        if len(far_persons_list):
            for far_persons, f in enumerate(far_persons_list):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # 좌표를 저장
                far_persons_region = frame[startY:endY, startX:endX]

                M = far_persons_region.shape[0]
                N = far_persons_region.shape[1]

                # 사진을 1/rati0 비율로 줄였다가 원래 사이즈로 확대하여 Pixel이 깨지도록 resize
                far_persons_region = cv2.resize(far_persons_region, None, fx=ratio, fy=ratio,
                                                interpolation=cv2.INTER_AREA)
                far_persons_region = cv2.resize(far_persons_region, (N, M), interpolation=cv2.INTER_AREA)

                # 원본 Frame에 다시 저장
                frame[startY:endY, startX:endX] = far_persons_region

        if len(near_persons_list):
            for near_persons, n in enumerate(near_persons_list):
                (startX, startY) = n[0], n[1]
                (endX, endY) = n[2], n[3]

                near_persons_region = frame[startY:endY, startX:endX]

                M = near_persons_region.shape[0]
                N = near_persons_region.shape[1]

                near_persons_region = cv2.resize(near_persons_region, None, fx=(ratio - 0.1), fy=(ratio - 0.1),
                                                 interpolation=cv2.INTER_AREA)
                near_persons_region = cv2.resize(near_persons_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = near_persons_region

        if len(move_list):
            for move, m in enumerate(move_list):
                (startX, startY) = m[0], m[1]
                (endX, endY) = m[2], m[3]

                move_region = frame[startY:endY, startX:endX]

                M = move_region.shape[0]
                N = move_region.shape[1]

                move_region = cv2.resize(move_region, None, fx=(ratio - 0.1), fy=(ratio - 0.1),
                                         interpolation=cv2.INTER_AREA)
                move_region = cv2.resize(move_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = move_region

        if len(rights_list):
            for rights, r in enumerate(rights_list):
                (startX, startY) = r[0], r[1]
                (endX, endY) = r[2], r[3]

                rights_region = frame[startY:endY, startX:endX]

                M = rights_region.shape[0]
                N = rights_region.shape[1]

                rights_region = cv2.resize(rights_region, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
                rights_region = cv2.resize(rights_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = rights_region

        if len(lefts_list):
            for lefts, l in enumerate(lefts_list):
                (startX, startY) = l[0], l[1]
                (endX, endY) = l[2], l[3]

                lefts_region = frame[startY:endY, startX:endX]

                M = lefts_region.shape[0]
                N = lefts_region.shape[1]

                lefts_region = cv2.resize(lefts_region, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
                lefts_region = cv2.resize(lefts_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = lefts_region

        # Q버튼 입력 시 ROI실행
        if cv2.waitKey(33) == ord('q'):
            roi()

    # 모자이크 Resize오류 시 ROI실행
    except Exception as e:
        print(str(e))
        roi()

    # 좌료 값이 잘 저장되나 확인하기 위해서 frame에 grid 표시
    # for k in range(0, 14):
    #     cv2.line(frame, (k * 50, 0), (k * 50, 480), (120, 120, 120), 1, 1)
    # for k in range(0, 9):
    #     cv2.line(frame, (0, k * 50), (720, k * 50), (120, 120, 120), 1, 1)

    # Video 보여주고 저장
    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(27) == ord('w'):
        exit()

capture.release()
out.release()
cv2.destroyAllWindows()

