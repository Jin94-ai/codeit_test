import albumentations as A
import cv2 
from albumentations.pytorch import ToTensorV2

TARGET_IMAGE_SIZE = (640, 640) 

# --- 1. 보수적인 훈련 데이터 증강 파이프라인 (Conservative) ---
# 알약의 핵심 특징을 최대한 보존하면서 강건성을 확보하는, 가장 기본적인 파이프라인.
# [적용된 주요 증강] 좌우반전, 약한 회전, 약한 이동, 밝기/대비, 약한 노이즈/블러, 미세 색조/채도.
def get_train_transforms_conservative(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        # --- 기하학적 변형 (반드시 적용 권장) ---
        A.HorizontalFlip(p=0.5), # 좌우 반전
        A.Rotate(limit=30, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # ±30도 범위 회전 (각인 보존)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # 약한 이동 (shift_limit=0.05), 스케일/회전은 0으로 설정하여 중복 방지

        # --- 색상 및 노이즈 변형 (반드시 적용 권장, 색조/채도 미세 조절) ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3), # 밝기/대비
        A.GaussNoise(p=0.1), # 약한 가우시안 노이즈
        A.Blur(blur_limit=3, p=0.1), # 약한 블러
        A.MotionBlur(blur_limit=3, p=0.1), # 약한 모션 블러
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=20, p=0.3), # 채도/색조 (매우 낮은 강도)
        A.RandomGamma(p=0.2), # 감마 조절

        # --- 이미지 크기 조정 및 정규화 (필수 전처리, 순서 중요) ---
        A.LongestMaxSize(max_size=max(target_size), p=1.0), # 가장 긴 축을 target_size에 맞게 조절
        A.PadIfNeeded( # target_size에 맞게 패딩 추가
            min_height=target_size[1],
            min_width=target_size[0],
            border_mode=cv2.BORDER_CONSTANT, # 경계는 상수 값 (검은색)
            value=(0,0,0), # 검은색으로 패딩
            p=1.0
        ),
        A.Normalize( # 표준화 (평균 0, 표준편차 1, 픽셀값 0-1 스케일)
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0) # PyTorch 텐서로 변환
    ], 
    bbox_params=A.BboxParams( # 바운딩 박스 변환 설정
        format='yolo', # YOLO 포맷 (x_center, y_center, w, h 정규화)
        label_fields=['class_labels'], # 클래스 라벨 필드 지정
        min_area=1.0, min_visibility=0.0, clip=True, min_width=1, min_height=1 # 유효성 검사
    ))


# --- 2. 균형 잡힌 훈련 데이터 증강 파이프라인 (Balanced) ---
# Conservative 파이프라인에 신중하게 적용할 기법들과 가려짐/압축 손실 시뮬레이션 기법들을 추가하여 성능을 더 끌어올림.
# [적용된 주요 증강] Conservative 기법 + 확대/축소, 컷아웃, JPEG 압축
def get_train_transforms_balanced(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        # --- 기하학적 변형 ---
        A.HorizontalFlip(p=0.5), # 좌우 반전
        A.Rotate(limit=60, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # ±60도 범위 회전 (Conservative보다 확장)
        A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=0, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # 약한 이동 (shift_limit=0.075), 확대/축소 (scale_limit=0.15)
        
        # --- 색상 및 노이즈 변형 ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=20, p=0.3),
        A.RandomGamma(p=0.2),
        
        # --- 가려짐 및 파일 손상 시뮬레이션 (신중하게 적용) ---
        A.CoarseDropout(max_holes=1, max_height=0.1, max_width=0.1, min_holes=1, fill_value=0, p=0.1), # 컷아웃 (부분 가려짐)
        A.JpegCompression(quality_lower=70, quality_upper=90, p=0.1), # JPEG 압축 손실

        # --- 고정 전처리 (순서 중요) ---
        A.LongestMaxSize(max_size=max(target_size), p=1.0), 
        A.PadIfNeeded(
            min_height=target_size[1], min_width=target_size[0], border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0
        ),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_area=1.0, min_visibility=0.0, clip=True, min_width=1, min_height=1
    ))


# --- 3. 공격적인/실험적인 훈련 데이터 증강 파이프라인 (Aggressive) ---
# Balanced 파이프라인에 더욱 다양한 기법 (원근, 전단, 부분 왜곡 등)을 추가하여 잠재적 성능 향상을 탐색.
# 강도가 높으므로 오버피팅 또는 성능 저하 가능성도 있으니 주의 깊은 모니터링이 필요.
# [적용된 주요 증강] Balanced 기법 + 원근 변환, 전단, 부분 왜곡, 증강 강도/확률 전반적 상향
def get_train_transforms_aggressive(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        # --- 기하학적 변형 (강도 및 종류 증가) ---
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # ±90도까지 회전 범위 확장
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), # 이동/확대/축소 강도 및 확률 증가
                           interpolation=cv2.INTER_LINEAR), # 보간법 지정
        A.Perspective(scale=(0.075, 0.15), p=0.2, pad_val=0), # 원근 변환 추가 (강도 및 확률 증가)
        A.PiecewiseAffine(scale=(0.01, 0.03), nb_rows=4, nb_cols=4, p=0.1), # 부분 왜곡 추가 (매우 약하게 시작)
        A.VerticalFlip(p=0.05), # 신중한 상하 반전 (각인 중요하지 않은 경우, 낮은 확률로 시도)
        
        # --- 색상 및 노이즈 변형 (강도 및 확률 증가) ---
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4), # 밝기/대비 강도 및 확률 증가
        A.GaussNoise(p=0.15), # 노이즈 확률 증가
        A.Blur(blur_limit=5, p=0.15), # 블러 강도 및 확률 증가
        A.MotionBlur(blur_limit=5, p=0.15),
        A.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=15, val_shift_limit=25, p=0.4), # 채도/색조 강도 및 확률 약간 증가
        A.RandomGamma(p=0.3),
        
        # --- 가려짐 및 파일 손상 시뮬레이션 (강도 및 확률 증가) ---
        A.CoarseDropout(max_holes=2, max_height=0.15, max_width=0.15, min_holes=1, fill_value=0, p=0.15), # 컷아웃 강도/확률 증가
        A.JpegCompression(quality_lower=60, quality_upper=85, p=0.15), # JPEG 압축 강도/확률 증가

        # --- 고정 전처리 (순서 중요) ---
        A.LongestMaxSize(max_size=max(target_size), p=1.0),
        A.PadIfNeeded(
            min_height=target_size[1], min_width=target_size[0], border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0
        ),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_area=1.0, min_visibility=0.0, clip=True, min_width=1, min_height=1
    ))


# --- 4. 검증 데이터셋 변환 파이프라인 ---
# 모델 입력에 필요한 기본적인 전처리(크기 조정, 정규화, 텐서 변환)만 수행.
def get_val_transforms(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=max(target_size), p=1.0),
        A.PadIfNeeded(
            min_height=target_size[1],
            min_width=target_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0,0,0), 
            p=1.0
        ),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams( # 검증 데이터셋에서도 바운딩 박스 형식과 유효성 검사는 필요.
        format='yolo', 
        label_fields=['class_labels'],
        min_area=1.0,
        min_visibility=0.0,
        clip=True,
        min_width=1,
        min_height=1
    ))