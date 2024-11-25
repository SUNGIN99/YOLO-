import cv2
import numpy as np
import pytesseract
import sys, os
import json 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_license_plate(image):
    # 1. 컬러 영상을 GrayScale 영상으로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거를 위한 Gaussian Blurring과 Morphology 연산 수행
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuring_element)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)
    enhanced = cv2.add(gray, top_hat)
    enhanced = cv2.subtract(enhanced, black_hat)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 2)
    
    # 3. Adaptive Thresholding(적응형 이진화) 수행
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 3)

    # 4. Edge 검출 및 Contour 그리기
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    contour_result = np.zeros((height, width, 1), dtype=np.uint8)
    cv2.drawContours(contour_result, contours, -1, (255, 255, 255))

    # 5. 번호판 글자로 추정되는 객체만 분류
    cnt_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / h
        
        # 번호판 특성 조건
        if area > 100 and w > 4 and h > 2 and 0.25 < ratio < 0.9:
            cnt_dict.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + (w / 2), 'cy': y + (h / 2)})

    # 6. 객체 간의 거리가 일정한 간격으로 연속되었다면 번호판으로 판단
    def find_character_contours(cnt_list):
        result = []
        for item1 in cnt_list:
            matches = []
            for item2 in cnt_list:
                if item1['contour'] is item2['contour']:  # 동일한 contour인 경우만 제외
                    continue

                dx = abs(item1['cx'] - item2['cx'])
                dy = abs(item1['cy'] - item2['cy'])
                distance = np.linalg.norm(np.array([item1['cx'], item1['cy']]) - np.array([item2['cx'], item2['cy']]))
                diag_len = np.sqrt(item1['w'] ** 2 + item1['h'] ** 2)

                # 거리 및 각도 조건
                angle = 90 if dx == 0 else np.degrees(np.arctan(dy / dx))
                if distance < diag_len * 5 and angle < 12:
                    matches.append(item2)
            
            matches.append(item1)

            if len(matches) < 5:  # 최소 글자 수
                continue

            result.append(matches)
        return result

    match_results = find_character_contours(cnt_dict)

    # 7. 가장 왼쪽 컨투어의 좌상단 좌표와 가장 오른쪽 컨투어의 우하단 좌표 추출
    plate_candidates = []
    for items in match_results:
        items = sorted(items, key=lambda x: x['cx'])  # x축 기준 정렬
        x_min = min([item['x'] for item in items])
        y_min = min([item['y'] for item in items])
        x_max = max([item['x'] + item['w'] for item in items])
        y_max = max([item['y'] + item['h'] for item in items])
        plate_candidates.append((x_min, y_min, x_max, y_max))

    # 8. Rectangle 그리기
    for (x_min, y_min, x_max, y_max) in plate_candidates:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 9. 해당 영역을 Crop하고 OCR 수행
    plate_texts = []
    for (x_min, y_min, x_max, y_max) in plate_candidates:
        cropped_plate = gray[y_min:y_max, x_min:x_max]
        ocr_result = pytesseract.image_to_string(cropped_plate, config="--psm 6 --oem 3 -c tessedit_char_whitelist=가나다라마바사아자차카타파하0123456789")  # 번호판 인식
        plate_texts.append(ocr_result.strip())

    return image, set(plate_texts)

# 테스트
if __name__ == "__main__":
    input_image_path = sys.argv[1]  # 차량 이미지 경로
    output_image_path = "result.jpg"

    jsonData = {"results" : []}

    for filename in os.listdir(input_image_path):
        file_path = os.path.join(input_image_path, filename)
        if os.path.isfile(file_path):  # 파일인지 확인

            img = cv2.imread(file_path)
            result_img, plates = detect_license_plate(img)

            #print("Detected Plates:", plates)
            jsonData["results"].append(list(plates))

    sys.stdout.write(json.dumps(jsonData))
    sys.stdout.flush()
    ## cv2.imwrite(output_image_path, result_img)
    ## cv2.imshow("Detected Plates", result_img)
    ## cv2.waitKey(0)
    ## cv2.destroyAllWindows()
